import { trace, context, SpanStatusCode, type Span } from "@opentelemetry/api";
import { NodeTracerProvider, BatchSpanProcessor } from "@opentelemetry/sdk-trace-node";
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

export interface TurnSpanHandle {
  startToolSpan(
    toolCallId: string,
    toolName: string,
    toolParams: unknown,
    isSubagent?: boolean,
  ): void;
  endToolSpan(toolCallId: string, output?: unknown, isError?: boolean, toolName?: string): void;
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

export function initTelemetry(logger?: Logger): TelemetryService | null {
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

    const exporter = new OTLPTraceExporter({
      url: tracesEndpoint,
      headers,
      timeoutMillis: 1500,
    });

    const batchProcessor = new BatchSpanProcessor(exporter, {
      maxQueueSize: 2048,
      maxExportBatchSize: 128,
      scheduledDelayMillis: 5000,
      exportTimeoutMillis: 1500,
    });

    const resource = resourceFromAttributes({
      [ATTR_SERVICE_NAME]: serviceName,
      "openinference.project.name": projectName,
    });

    const provider = new NodeTracerProvider({
      resource,
      spanProcessors: [batchProcessor],
      spanLimits: {
        attributeValueLengthLimit: 8192,
      },
    });

    provider.register();
    const tracer = trace.getTracer("claude-agent-acp", getPackageVersion());
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
        const activeToolSpans = new Map<string, Span>();
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

            activeToolSpans.set(toolCallId, toolSpan);
          },

          endToolSpan(
            toolCallId: string,
            output?: unknown,
            isError?: boolean,
            toolName?: string,
          ): void {
            const toolSpan = activeToolSpans.get(toolCallId);
            if (!toolSpan) {
              if (toolName) {
                rootSpan.addEvent("unmapped_tool_result", {
                  "tool.call_id": toolCallId,
                  "tool.name": toolName,
                });
              }
              return;
            }

            // Invariant 5: Privacy-First Metadata-Only
            const outputBytes = computeOutputBytes(output);
            toolSpan.setAttribute("tool.output_bytes", outputBytes);

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
            for (const orphanSpan of activeToolSpans.values()) {
              orphanSpan.setStatus({
                code: SpanStatusCode.ERROR,
                message: "orphaned_at_turn_end",
              });
              orphanSpan.setAttribute("error.message", "orphaned_at_turn_end");
              orphanSpan.setAttribute("tool.is_error", true);
              orphanSpan.end();
            }
            activeToolSpans.clear();

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
