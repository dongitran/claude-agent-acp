import { trace, context, SpanStatusCode, type Span } from "@opentelemetry/api";
import {
  NodeTracerProvider,
  BatchSpanProcessor,
  type SpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import type { Logger } from "./acp-agent.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// Secret sanitization patterns
const PREFIXED_PATTERNS = [
  /nsec1[0-9a-z]{58}/gi, // with /i flag against Bech32 uppercase evasion
  /sk-ant-[0-9a-zA-Z_.-]+/g,
  /\bgl(pat|rt|dt|ptt|cbt|oas)-[0-9a-zA-Z_-]+/g,
  /\bgh[pousr]_[0-9a-zA-Z]{36}\b/g,
  /ya29\.[0-9a-zA-Z_-]+/g,
  /1\/\/[0-9a-zA-Z_-]+/g,
  /\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+/g,
];

let cachedProcessSecrets: string[] | null = null;

export function _resetProcessSecretsCache(): void {
  cachedProcessSecrets = null;
}

function getProcessSecrets(): string[] {
  if (cachedProcessSecrets !== null) {
    return cachedProcessSecrets;
  }
  const secrets: string[] = [];
  const sensitiveKeys = [
    "ANTHROPIC_API_KEY",
    "BUZZ_PRIVATE_KEY",
    "GITLAB_TOKEN",
    "AGY_OAUTH_TOKEN_B64",
    "PHOENIX_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "MEETING_GATE_TOKEN",
  ];

  for (const key of sensitiveKeys) {
    const val = process.env[key];
    if (val && typeof val === "string" && val.length >= 8) {
      secrets.push(val);
      try {
        const b64 = Buffer.from(val).toString("base64");
        if (b64.length >= 8) {
          secrets.push(b64);
        }
      } catch {
        // ignore
      }
    }
  }

  // Deduplicate and sort descending by length to replace longer strings first
  cachedProcessSecrets = Array.from(new Set(secrets)).sort((a, b) => b.length - a.length);
  return cachedProcessSecrets;
}

export function redactSecrets(input: unknown): string {
  if (input === null || input === undefined) {
    return "";
  }
  let text = typeof input === "string" ? input : JSON.stringify(input);

  // 1. Exact match against known process secrets (and base64 encodings)
  const processSecrets = getProcessSecrets();
  for (const secret of processSecrets) {
    if (text.includes(secret)) {
      text = text.split(secret).join("[REDACTED]");
    }
  }

  // 2. Prefixed patterns (Nostr, Anthropic, GitLab, GitHub, Google OAuth, JWT)
  for (const pattern of PREFIXED_PATTERNS) {
    text = text.replace(pattern, "[REDACTED]");
  }

  return text;
}

export function truncatePayload(text: string, maxLen = 8192): string {
  if (!text) {
    return "";
  }
  if (text.length <= maxLen) {
    return text;
  }
  return text.slice(0, maxLen) + "... [TRUNCATED]";
}

/**
 * P0 Defense: Computes byte length of output without throwing TypeError
 * when output is an Array of ContentBlocks, an Object, a Buffer, or a String.
 */
export function computeOutputBytes(output: unknown): number {
  if (output === null || output === undefined) return 0;
  if (typeof output === "string") return Buffer.byteLength(output, "utf8");
  if (Buffer.isBuffer(output)) return output.length;
  try {
    return Buffer.byteLength(JSON.stringify(output), "utf8");
  } catch {
    try {
      return Buffer.byteLength(String(output), "utf8");
    } catch {
      return 0;
    }
  }
}

export interface AnthropicUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  thinking_tokens?: number | null;
}

/**
 * Robust differential cost tracker that gracefully handles session compaction
 * and resets without causing zero or negative turn costs.
 */
export class RobustDifferentialCostTracker {
  private lastCostBySession = new Map<string, number>();

  public computeTurnCost(sessionId: string, cumulativeCostUsd: number | undefined): number {
    if (
      typeof cumulativeCostUsd !== "number" ||
      isNaN(cumulativeCostUsd) ||
      cumulativeCostUsd <= 0
    ) {
      return 0;
    }
    const lastCost = this.lastCostBySession.get(sessionId) ?? 0;
    let delta: number;

    // If cumulative cost drops below last recorded baseline (session reset/compaction), use cumulativeCostUsd
    if (cumulativeCostUsd < lastCost) {
      delta = cumulativeCostUsd;
    } else {
      delta = cumulativeCostUsd - lastCost;
    }

    this.lastCostBySession.set(sessionId, cumulativeCostUsd);
    return Math.max(0, Number(delta.toFixed(6)));
  }

  public purgeSession(sessionId: string): void {
    this.lastCostBySession.delete(sessionId);
  }

  public clear(): void {
    this.lastCostBySession.clear();
  }
}

export interface NostrMetadata {
  threadRootId?: string;
  eventId?: string;
  channelId?: string;
  channelName?: string;
  replyToId?: string;
  senderName?: string;
  senderPubkey?: string;
}

/**
 * Hardened regex parser extracting Nostr context envelope.
 * Isolates the active envelope using lastIndexOf("Event ID:") to prevent
 * capturing cancelled messages in steered batches.
 */
export function parseNostrContext(rawText: string): NostrMetadata {
  const result: NostrMetadata = {};
  if (!rawText) return result;

  // Defensive isolation: isolate last envelope in steered/cancelled batch
  const lastEventIdx = rawText.lastIndexOf("Event ID:");
  let text = rawText;
  if (lastEventIdx !== -1) {
    const lastChannelIdx = rawText.lastIndexOf("Channel:", lastEventIdx);
    if (lastChannelIdx !== -1) {
      text = rawText.substring(lastChannelIdx);
    } else {
      const lastDoubleNewline = rawText.lastIndexOf("\n\n", lastEventIdx);
      text = lastDoubleNewline !== -1 ? rawText.substring(lastDoubleNewline) : rawText;
    }
  }

  // 1. Thread Root ID: "Thread root: <64-hex>" or "root=<64-hex>"
  const rootMatch =
    text.match(/Thread root:\s*([a-f0-9]{64})/i) || text.match(/\broot=([a-f0-9]{64})\b/i);
  if (rootMatch) {
    result.threadRootId = rootMatch[1].toLowerCase();
  }

  // 2. Event ID: "Event ID: <64-hex>"
  const eventMatch = text.match(/Event ID:\s*([a-f0-9]{64})/i);
  if (eventMatch) {
    result.eventId = eventMatch[1].toLowerCase();
    if (!result.threadRootId) {
      result.threadRootId = result.eventId;
    }
  }

  // 3. Channel UUID: "(#<UUID>)" or "Channel: <UUID>" (DM Scope)
  const channelMatch =
    text.match(/#([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})/i) ||
    text.match(/Channel:\s*([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})/i);
  if (channelMatch) {
    result.channelId = channelMatch[1].toLowerCase();
  }

  // 4. Channel Name: "Channel: <name> (#...)"
  const channelNameMatch = text.match(/Channel:\s*([a-zA-Z0-9_-]+)\s*(?:\(|$)/i);
  const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
    channelNameMatch?.[1] ?? "",
  );
  if (channelNameMatch && !isUuid) {
    result.channelName = channelNameMatch[1];
  }

  // 5. Parent / Reply-To ID: "parent=<64-hex>"
  const parentMatch = text.match(/\bparent=([a-f0-9]{64})\b/i);
  if (parentMatch) {
    result.replyToId = parentMatch[1].toLowerCase();
  }

  // 6. Sender Info: supports parentheses in names and Unicode/emojis
  const fromMatch = text.match(/From:\s*(.+?)\s*\((?:[^)]*?hex:\s*([a-f0-9]{64}))[^)]*\)/i);
  if (fromMatch) {
    result.senderName = fromMatch[1].trim();
    result.senderPubkey = fromMatch[2].toLowerCase();
  }

  return result;
}

/** What stands in `From:` when the sender has no display name: `npub1… (hex: …)`. */
const BARE_IDENTITY = /^(?:npub1[0-9a-z]+|[0-9a-f]{64})$/i;

/** Most of a steered message that is read for its ids: buzz-acp's block is far smaller than this. */
const STEERED_PARSE_CHARS = 262144;

/**
 * Reads the ids of a message steered into a running turn: one event block, as buzz-acp renders it.
 *
 * It differs from `parseNostrContext` in two ways. The block starts at its first `Event ID:` line and its header (the
 * lines before `Content:`) is read from there, so that a message that quotes `Event ID: …` in its own text cannot
 * name another event; `Parsed:` is the block's last such line, after the text, for the same reason. And a steered
 * message is rendered without the profile lookup the opening prompt gets, so `From:` usually reads `npub1… (hex: …)`:
 * a sender name is kept only when the block has one.
 */
export function parseSteeredMessage(rawText: string): NostrMetadata {
  const result: NostrMetadata = {};
  const text = rawText.slice(0, STEERED_PARSE_CHARS);
  const start = text.search(/^Event ID:/m);
  if (start === -1) return result;
  const block = text.slice(start);
  const contentAt = block.search(/^Content:/m);
  const header = contentAt === -1 ? block : block.slice(0, contentAt);

  const eventId = /^Event ID:[ \t]*([a-f0-9]{64})[ \t]*$/im.exec(header);
  if (eventId) result.eventId = eventId[1].toLowerCase();

  const channelId =
    /^Channel:[ \t]*(?:[^\n]*\(#)?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\)?[ \t]*$/im.exec(
      header,
    );
  if (channelId) result.channelId = channelId[1].toLowerCase();

  const from =
    /^From:[ \t]*(.+?)[ \t]*\((?:[^)\n]*?hex:[ \t]*([a-f0-9]{64}))[^)\n]*\)[ \t]*$/im.exec(header);
  if (from) {
    result.senderPubkey = from[2].toLowerCase();
    const name = from[1].trim();
    if (name && !BARE_IDENTITY.test(name)) result.senderName = name;
  }

  const parsed = [...block.matchAll(/^Parsed:[ \t]*(.*)$/gm)].pop()?.[1] ?? "";
  const parent = /\bparent=([a-f0-9]{64})\b/i.exec(parsed);
  if (parent) result.replyToId = parent[1].toLowerCase();
  const root = /\broot=([a-f0-9]{64})\b/i.exec(parsed);
  // A message with no root of its own starts its thread, as the opening prompt's parser has it
  if (root) result.threadRootId = root[1].toLowerCase();
  else if (result.eventId) result.threadRootId = result.eventId;

  return result;
}

/** Tools whose result names the Buzz message they posted: `posted (N characters) · event_id <64 hex>`. */
const POSTED_EVENT_TOOLS = new Set(["mcp__buzz-thread__reply"]);

/** Longest part of a tool result searched for the posted id; the id is at its start. */
const POSTED_EVENT_SEARCH_CHARS = 4096;

/** Most `steer` spans one turn records, so that a channel flooding an agent with mentions cannot flood the trace. */
export const MAX_STEER_SPANS_PER_TURN = 200;

/** The text blocks of a tool result: a string, a content block, or a list of blocks. */
function toolResultTexts(output: unknown, depth = 0): string[] {
  if (typeof output === "string") return [output];
  if (output === null || typeof output !== "object" || depth > 2) return [];
  if (Array.isArray(output)) return output.flatMap((block) => toolResultTexts(block, depth + 1));
  const record = output as Record<string, unknown>;
  const texts: string[] = [];
  if (typeof record.text === "string") texts.push(record.text);
  if (record.content !== undefined) texts.push(...toolResultTexts(record.content, depth + 1));
  return texts;
}

/**
 * The id of the Buzz message a posting tool returned: the 64 lowercase hex characters after `event_id `, and nothing
 * else of the result. Undefined when the result holds no such id.
 */
export function extractPostedEventId(output: unknown): string | undefined {
  for (const text of toolResultTexts(output)) {
    const match = /\bevent_id ([0-9a-f]{64})(?![0-9a-f])/.exec(
      text.slice(0, POSTED_EVENT_SEARCH_CHARS),
    );
    if (match) return match[1];
  }
  return undefined;
}

/** A message steered into a running turn (see `TurnSpanHandle.startSteerSpan`). */
export interface SteerSpanInput {
  /** The uuid the steered message was pushed to the SDK with. Its echo, or a result that names it, ends the span. */
  uuid: string;
  /** The steered message as the agent received it. Redacted and truncated here, as a turn's prompt is. */
  text: string;
  /** How the SDK was told to deliver it: `later` while a permission or elicitation request is open. */
  priority: "now" | "later";
  nostrMeta?: NostrMetadata;
  /** When the steer arrived, in milliseconds since the epoch: where the span starts. */
  arrivedAt?: number;
}

/** What ended a `steer` span before the turn did: the SDK's echo of the message, or a result that names it. */
export type SteerDelivery = "echo" | "result";

export interface TurnSpanHandle {
  startToolSpan(
    toolCallId: string,
    toolName: string,
    toolParams: unknown,
    isSubagent?: boolean,
  ): void;
  endToolSpan(toolCallId: string, output?: unknown, isError?: boolean, toolName?: string): void;
  /**
   * Records a message injected into this turn as a `steer` child span of the turn's span. The span stays open until
   * `endSteerSpan` names it, or until the turn ends (`end` closes it with `steer.delivered=false`). A no-op once the
   * turn has ended. Never throws: tracing must not change how a steer is delivered.
   */
  startSteerSpan(steer: SteerSpanInput): void;
  /** The SDK took the steered message: ends its span with `steer.delivered=true`. Unknown or ended uuids are ignored. */
  endSteerSpan(uuid: string, delivery: SteerDelivery): void;
  recordUsage(usage: AnthropicUsage, totalCostUsd?: number): void;
  end(
    status: "OK" | "ERROR" | "CANCELLED",
    outputOrError?: string,
    failureKind?: string,
    errorMessage?: string,
  ): void;
}

export interface TelemetryService {
  startTurnSpan(
    sessionId: string,
    promptText: string,
    modelName?: string,
    nostrMeta?: NostrMetadata,
  ): TurnSpanHandle;
  backgroundFlush(timeoutMs?: number): void;
  shutdown(): Promise<void>;
}

export type TelemetryEngine = TelemetryService;

// Single-Flight Coalescing flush state
let inFlightFlush: Promise<void> | null = null;
let exportErrorCount = 0;

function logExportError(logger: Logger | undefined, err: unknown): void {
  exportErrorCount++;
  if (exportErrorCount === 1 || exportErrorCount % 100 === 0) {
    const msg = err instanceof Error ? err.message : String(err);
    logger?.error(`[Telemetry] Trace export failed (occurrence #${exportErrorCount}): ${msg}`);
  }
}

/**
 * Coalesced non-blocking background flush with single-flight guarantee
 * and unref timer to avoid pinning the Node.js event loop.
 */
export function backgroundFlush(
  provider: { forceFlush(): Promise<void> },
  logger?: Logger,
  timeoutMs = 1500,
): Promise<void> {
  if (inFlightFlush) {
    return inFlightFlush;
  }

  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutPromise = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, timeoutMs);
    timer.unref(); // Prevent timer from keeping event loop active
  });

  inFlightFlush = Promise.race([
    provider.forceFlush().catch((err: unknown) => {
      logExportError(logger, err);
    }),
    timeoutPromise,
  ]).finally(() => {
    if (timer) clearTimeout(timer);
    inFlightFlush = null;
  });

  return inFlightFlush;
}

export function getPackageVersion(): string {
  try {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const pkgPath = join(__dirname, "../package.json");
    const pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
    return pkg.version || "0.100.0";
  } catch {
    return "0.100.0";
  }
}

export interface TelemetryInitOptions {
  /** Takes the finished spans in place of the OTLP exporter. For tests, which read the spans back. */
  spanProcessors?: SpanProcessor[];
}

export function initTelemetry(
  logger?: Logger,
  options: TelemetryInitOptions = {},
): TelemetryService | null {
  const tracesEndpoint =
    process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT || process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
  if (!tracesEndpoint) {
    return null;
  }

  try {
    const serviceName = process.env.OTEL_SERVICE_NAME || "claude-agent";
    const projectName = process.env.PHOENIX_PROJECT_NAME || "default";
    const apiKey = process.env.PHOENIX_API_KEY;

    const headers: Record<string, string> = {
      "x-project-name": projectName,
    };
    if (apiKey) {
      headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const spanProcessors = options.spanProcessors ?? [
      new BatchSpanProcessor(
        new OTLPTraceExporter({
          url: tracesEndpoint,
          headers,
          timeoutMillis: 1500,
        }),
        {
          maxQueueSize: 2048,
          maxExportBatchSize: 128,
          scheduledDelayMillis: 5000,
          exportTimeoutMillis: 1500,
        },
      ),
    ];

    const resource = resourceFromAttributes({
      [ATTR_SERVICE_NAME]: serviceName,
      "openinference.project.name": projectName,
    });

    const provider = new NodeTracerProvider({
      resource,
      spanProcessors,
      spanLimits: {
        attributeValueLengthLimit: 8192,
      },
    });

    provider.register();
    // From this provider, not the global one: a second provider in one process (a test's) does not replace the first
    // global registration, and its spans would go to the wrong place.
    const tracer = provider.getTracer("claude-agent-acp", getPackageVersion());
    const costTracker = new RobustDifferentialCostTracker();

    return {
      startTurnSpan(
        sessionId: string,
        promptText: string,
        modelName?: string,
        nostrMeta?: NostrMetadata,
      ): TurnSpanHandle {
        // CPU Optimization: truncate to 32k before running regex redaction
        const preTruncated = truncatePayload(promptText, 32768);
        const sanitizedPrompt = truncatePayload(redactSecrets(preTruncated), 8192);
        const resolvedSessionId = nostrMeta?.threadRootId || sessionId;

        const rootSpan = tracer.startSpan("agent_turn", {
          attributes: {
            "openinference.span.kind": "AGENT",
            "session.id": resolvedSessionId,
            "user.id": nostrMeta?.senderPubkey || "anonymous",
            "agent.name": serviceName,
            "input.value": sanitizedPrompt,
            "input.mime_type": "text/plain",
            ...(nostrMeta?.threadRootId && { "nostr.thread_root_id": nostrMeta.threadRootId }),
            ...(nostrMeta?.eventId && { "nostr.event_id": nostrMeta.eventId }),
            ...(nostrMeta?.channelId && { "nostr.channel_id": nostrMeta.channelId }),
            ...(nostrMeta?.channelName && { "nostr.channel_name": nostrMeta.channelName }),
            ...(nostrMeta?.replyToId && { "nostr.reply_to_id": nostrMeta.replyToId }),
            ...(nostrMeta?.senderName && { "nostr.sender_name": nostrMeta.senderName }),
          },
        });

        const rootContext = trace.setSpan(context.active(), rootSpan);
        const effectiveModel = modelName || process.env.ANTHROPIC_MODEL || "sonnet";

        const llmSpan = tracer.startSpan(
          `llm:${effectiveModel}`,
          {
            attributes: {
              "openinference.span.kind": "LLM",
              "session.id": resolvedSessionId,
              "llm.model_name": effectiveModel,
              "llm.provider": "anthropic",
              "input.value": sanitizedPrompt,
              "input.mime_type": "text/plain",
            },
          },
          rootContext,
        );

        // Keyed by Anthropic immutable tool_use_id (toolu_*)
        const activeToolSpans = new Map<string, { span: Span; toolName: string }>();
        // Messages steered into this turn whose span is still open, keyed by the uuid the SDK got them with
        const openSteerSpans = new Map<string, Span>();
        let steersSeen = 0;
        let hasEnded = false;

        return {
          startToolSpan(
            toolCallId: string,
            toolName: string,
            toolParams: unknown,
            isSubagent = false,
          ): void {
            if (activeToolSpans.has(toolCallId)) {
              return;
            }
            const preTruncParams = truncatePayload(
              typeof toolParams === "string" ? toolParams : JSON.stringify(toolParams),
              32768,
            );
            const sanitizedParams = truncatePayload(redactSecrets(preTruncParams), 8192);

            const toolSpan = tracer.startSpan(
              isSubagent ? `agent:${toolName}` : `tool:${toolName}`,
              {
                attributes: {
                  "openinference.span.kind": isSubagent ? "AGENT" : "TOOL",
                  "session.id": resolvedSessionId,
                  "tool.name": toolName,
                  "tool.call_id": toolCallId,
                  "tool.parameters": sanitizedParams,
                  "input.value": sanitizedParams,
                  "input.mime_type": "application/json",
                  ...(isSubagent && { "agent.name": toolName }),
                },
              },
              rootContext,
            );

            activeToolSpans.set(toolCallId, { span: toolSpan, toolName });
          },

          endToolSpan(
            toolCallId: string,
            output?: unknown,
            isError?: boolean,
            toolName?: string,
          ): void {
            const active = activeToolSpans.get(toolCallId);
            if (!active) {
              if (toolName) {
                rootSpan.addEvent("unmapped_tool_result", {
                  "tool.call_id": toolCallId,
                  "tool.name": toolName,
                });
              }
              return;
            }
            const toolSpan = active.span;

            // Invariant 5: Privacy-First Metadata-Only
            const outputBytes = computeOutputBytes(output);
            toolSpan.setAttribute("tool.output_bytes", outputBytes);

            // The one thing lifted out of a result: the id of the Buzz message a posting tool says it posted, so a later
            // turn that replies to that message can be tied to this one. Only the id, never the result's text.
            if (!isError && POSTED_EVENT_TOOLS.has(active.toolName)) {
              const postedEventId = extractPostedEventId(output);
              if (postedEventId) {
                toolSpan.setAttribute("nostr.posted_event_id", postedEventId);
              }
            }

            if (process.env.OTEL_CAPTURE_TOOL_OUTPUT === "true" && output !== undefined) {
              const rawStr = typeof output === "string" ? output : JSON.stringify(output);
              const preTruncOut = truncatePayload(rawStr, 32768);
              toolSpan.setAttribute(
                "output.value",
                truncatePayload(redactSecrets(preTruncOut), 8192),
              );
              toolSpan.setAttribute("output.mime_type", "text/plain");
            }

            toolSpan.setAttribute("tool.is_error", Boolean(isError));
            if (isError) {
              toolSpan.setStatus({
                code: SpanStatusCode.ERROR,
                message: "tool_execution_failed",
              });
            } else {
              toolSpan.setStatus({ code: SpanStatusCode.OK });
            }

            toolSpan.end();
            activeToolSpans.delete(toolCallId);
          },

          startSteerSpan(steer: SteerSpanInput): void {
            // A span started after the turn ended would never be closed.
            if (hasEnded || openSteerSpans.has(steer.uuid)) return;
            try {
              steersSeen++;
              if (steersSeen > MAX_STEER_SPANS_PER_TURN) {
                if (steersSeen === MAX_STEER_SPANS_PER_TURN + 1) {
                  rootSpan.addEvent("steer_spans_capped", { "steer.span_cap": steersSeen - 1 });
                }
                return;
              }
              // The same redaction and cut as the turn's own prompt (see startTurnSpan above)
              const preTruncated = truncatePayload(steer.text, 32768);
              const sanitizedText = truncatePayload(redactSecrets(preTruncated), 8192);
              const meta = steer.nostrMeta;
              const steerSpan = tracer.startSpan(
                "steer",
                {
                  ...(steer.arrivedAt !== undefined && { startTime: new Date(steer.arrivedAt) }),
                  attributes: {
                    "openinference.span.kind": "CHAIN",
                    "session.id": resolvedSessionId,
                    "input.value": sanitizedText,
                    "input.mime_type": "text/plain",
                    "steer.priority": steer.priority,
                    "steer.uuid": steer.uuid,
                    ...(meta?.eventId && { "nostr.event_id": meta.eventId }),
                    ...(meta?.senderPubkey && { "nostr.sender_pubkey": meta.senderPubkey }),
                    ...(meta?.senderName && { "nostr.sender_name": meta.senderName }),
                    ...(meta?.replyToId && { "nostr.reply_to_id": meta.replyToId }),
                    ...(meta?.threadRootId && { "nostr.thread_root_id": meta.threadRootId }),
                    ...(meta?.channelId && { "nostr.channel_id": meta.channelId }),
                  },
                },
                rootContext,
              );
              openSteerSpans.set(steer.uuid, steerSpan);
            } catch (err) {
              logger?.error(
                `[Telemetry] Could not start a steer span: ${err instanceof Error ? err.message : String(err)}`,
              );
            }
          },

          endSteerSpan(uuid: string, delivery: SteerDelivery): void {
            const steerSpan = openSteerSpans.get(uuid);
            if (!steerSpan) return;
            openSteerSpans.delete(uuid);
            try {
              steerSpan.setAttributes({ "steer.delivered": true, "steer.end_reason": delivery });
              steerSpan.setStatus({ code: SpanStatusCode.OK });
              steerSpan.end();
            } catch (err) {
              logger?.error(
                `[Telemetry] Could not end a steer span: ${err instanceof Error ? err.message : String(err)}`,
              );
            }
          },

          recordUsage(usage: AnthropicUsage, cumulativeCostUsd?: number): void {
            const promptTokens =
              (usage.input_tokens || 0) +
              (usage.cache_read_input_tokens || 0) +
              (usage.cache_creation_input_tokens || 0);
            const completionTokens = usage.output_tokens || 0; // Already includes thinking_tokens
            const totalTokens = promptTokens + completionTokens;

            llmSpan.setAttributes({
              "llm.token_count.prompt": promptTokens,
              "llm.token_count.completion": completionTokens,
              "llm.token_count.total": totalTokens,
              ...(usage.cache_read_input_tokens !== undefined &&
                usage.cache_read_input_tokens !== null && {
                  "llm.token_count.prompt_details.cache_read": usage.cache_read_input_tokens,
                }),
              ...(usage.cache_creation_input_tokens !== undefined &&
                usage.cache_creation_input_tokens !== null && {
                  "llm.token_count.prompt_details.cache_write": usage.cache_creation_input_tokens,
                }),
              ...(usage.thinking_tokens !== undefined &&
                usage.thinking_tokens !== null && {
                  "llm.token_count.completion_details.reasoning": usage.thinking_tokens,
                }),
            });

            rootSpan.setAttributes({
              "llm.token_count.prompt": promptTokens,
              "llm.token_count.completion": completionTokens,
              "llm.token_count.total": totalTokens,
            });

            if (cumulativeCostUsd !== undefined) {
              const turnCost = costTracker.computeTurnCost(resolvedSessionId, cumulativeCostUsd);
              llmSpan.setAttribute("llm.cost.total", turnCost);
              rootSpan.setAttribute("llm.cost.total", turnCost);
            }
          },

          end(
            status: "OK" | "ERROR" | "CANCELLED",
            outputOrError?: string,
            failureKind?: string,
            errorMessage?: string,
          ): void {
            if (hasEnded) return;
            hasEnded = true;

            // Turn boundary sweep: close any orphaned tool spans safely
            for (const { span: orphanSpan } of activeToolSpans.values()) {
              orphanSpan.setStatus({
                code: SpanStatusCode.ERROR,
                message: "orphaned_at_turn_end",
              });
              orphanSpan.setAttribute("error.message", "orphaned_at_turn_end");
              orphanSpan.setAttribute("tool.is_error", true);
              orphanSpan.end();
            }
            activeToolSpans.clear();

            // A steered message the SDK never took before the turn ended (cancel, failure, a hand-off to the next
            // prompt, the session closing): its span ends with the turn, not delivered. Not an error of the turn.
            for (const steerSpan of openSteerSpans.values()) {
              try {
                steerSpan.setAttributes({
                  "steer.delivered": false,
                  "steer.end_reason": "turn_end",
                });
                steerSpan.end();
              } catch (err) {
                logger?.error(
                  `[Telemetry] Could not end a steer span: ${err instanceof Error ? err.message : String(err)}`,
                );
              }
            }
            openSteerSpans.clear();

            const outputValue = status === "OK" ? outputOrError : undefined;
            if (outputValue) {
              const preTruncOut = truncatePayload(outputValue, 32768);
              const sanitizedOut = truncatePayload(redactSecrets(preTruncOut), 8192);
              llmSpan.setAttribute("output.value", sanitizedOut);
              llmSpan.setAttribute("output.mime_type", "text/plain");
              rootSpan.setAttribute("output.value", sanitizedOut);
              rootSpan.setAttribute("output.mime_type", "text/plain");
            }

            if (status === "CANCELLED") {
              // Invariant 6: Status UNSET, record stop_reason to preserve SLO Error Budget
              rootSpan.setAttribute("claude.stop_reason", "cancelled");
              llmSpan.setAttribute("claude.stop_reason", "cancelled");
            } else if (status === "ERROR") {
              const errDesc =
                errorMessage ||
                failureKind ||
                (status === "ERROR" ? outputOrError : undefined) ||
                "turn_failed";
              rootSpan.setStatus({ code: SpanStatusCode.ERROR, message: errDesc });
              llmSpan.setStatus({ code: SpanStatusCode.ERROR, message: errDesc });
              if (failureKind) rootSpan.setAttribute("error.kind", failureKind);
              rootSpan.setAttribute("error.message", truncatePayload(errDesc, 1024));
            } else {
              rootSpan.setStatus({ code: SpanStatusCode.OK });
              llmSpan.setStatus({ code: SpanStatusCode.OK });
            }

            llmSpan.end();
            rootSpan.end();
          },
        };
      },

      backgroundFlush(timeoutMs = 1500): void {
        backgroundFlush(provider, logger, timeoutMs);
      },

      async shutdown(): Promise<void> {
        costTracker.clear();
        try {
          await Promise.race([
            provider.shutdown(),
            new Promise<void>((resolve) => setTimeout(resolve, 2000)),
          ]);
        } catch (err) {
          logExportError(logger, err);
        }
      },
    };
  } catch (err) {
    logger?.error(
      `[Telemetry] Failed to initialize OpenTelemetry SDK, running fail-open: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
    return null;
  }
}
