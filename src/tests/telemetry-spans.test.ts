import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  type ReadableSpan,
} from "@opentelemetry/sdk-trace-node";
import {
  MAX_STEER_SPANS_PER_TURN,
  extractPostedEventId,
  initTelemetry,
  parseNostrContext,
  parseSteeredMessage,
  type TelemetryService,
  type TurnSpanHandle,
} from "../telemetry.js";

const OWNER = "0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9";
const AGENT = "1f2e3d4c5b6a79881706f5e4d3c2b1a01f2e3d4c5b6a79881706f5e4d3c2b1a0";
const CHANNEL = "0b5c1d2e-3f40-4a5b-8c6d-7e8f90a1b2c3";
const ROOT_ID = "0000000000000000000000000000000000000000000000000000000000000a01";
const OPENING_ID = "0000000000000000000000000000000000000000000000000000000000000a02";
const STEERED_ID = "0000000000000000000000000000000000000000000000000000000000000a03";
const POSTED_ID = "0000000000000000000000000000000000000000000000000000000000000a04";

/** A turn's opening prompt as buzz-acp renders it, with the sender's display name. */
const OPENING = `[Buzz event: @mention]
Event ID: ${OPENING_ID}
Channel: support (#${CHANNEL})
Kind: 9
From: Alice Example (npub: npub1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq, hex: ${OWNER})
Time: 2026-11-08T04:20:00+00:00
Content: @Twin first question
Parsed: root=${ROOT_ID}, mentions=[${AGENT}]`;

/** A steered message as the fork receives it: no display name in `From:`. */
const STEERED = `[New message — arrived while you were working]

[Buzz event: @mention]
Event ID: ${STEERED_ID}
Channel: ${CHANNEL}
Kind: 9
From: npub1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq (hex: ${OWNER})
Time: 2026-11-08T04:20:04+00:00
Content: @Twin Separate question: which SAP table holds material descriptions? Table name only, plus one sentence.
Tags: [["h","${CHANNEL}"],["e","${ROOT_ID}","","reply"],["p","${AGENT}"]]
Parsed: parent=${ROOT_ID}, root=${ROOT_ID}, mentions=[${AGENT}]

Note: A new message arrived while you were working. Continue your in-progress work and incorporate the new message if it's relevant; if it's unrelated, you may briefly acknowledge it and carry on.`;

/** Milliseconds since the epoch of a span's start or end. */
const ms = (time: [number, number]) => time[0] * 1000 + time[1] / 1e6;

describe("Steered messages and posted ids in the trace", () => {
  const originalEnv = process.env;
  let exporter: InMemorySpanExporter;
  let service: TelemetryService;

  beforeEach(() => {
    process.env = {
      ...originalEnv,
      OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: "http://127.0.0.1:4318/v1/traces",
    };
    exporter = new InMemorySpanExporter();
    service = initTelemetry(undefined, { spanProcessors: [new SimpleSpanProcessor(exporter)] })!;
    expect(service).not.toBeNull();
  });

  afterEach(async () => {
    await service.shutdown();
    process.env = originalEnv;
  });

  const finished = (name: string): ReadableSpan[] =>
    exporter.getFinishedSpans().filter((span) => span.name === name);
  const only = (name: string): ReadableSpan => {
    const found = finished(name);
    expect(found, `spans named ${name}`).toHaveLength(1);
    return found[0];
  };
  const startTurn = (): TurnSpanHandle =>
    service.startTurnSpan("acp-session", OPENING, "sonnet", parseNostrContext(OPENING));
  const steer = (
    handle: TurnSpanHandle,
    uuid: string,
    text = STEERED,
    priority: "now" | "later" = "now",
  ) => handle.startSteerSpan({ uuid, text, priority, nostrMeta: parseSteeredMessage(text) });

  describe("parseSteeredMessage", () => {
    it("reads the ids of a steered block and leaves out a sender name that is only an npub", () => {
      const meta = parseSteeredMessage(STEERED);
      expect(meta).toMatchObject({
        eventId: STEERED_ID,
        senderPubkey: OWNER,
        replyToId: ROOT_ID,
        threadRootId: ROOT_ID,
        channelId: CHANNEL,
      });
      expect(meta.senderName).toBeUndefined();
      // The opening prompt's parser keeps what it always kept.
      expect(parseNostrContext(STEERED).senderName).toMatch(/^npub1/);
    });

    it("keeps a display name when the block has one", () => {
      const named = STEERED.replace(
        /From: npub1\S+ \(hex:/,
        "From: Alice Example (npub: npub1abc, hex:",
      );
      expect(parseSteeredMessage(named).senderName).toBe("Alice Example");
    });

    it("returns what it can for a block with no Buzz event", () => {
      expect(parseSteeredMessage("also check the totals")).toEqual({});
    });

    it("is not moved by what the message itself says: a quoted `Event ID:`, `From:` or `Parsed:` line", () => {
      const quoted = STEERED.replace(
        "Content: @Twin Separate question",
        `Content: @Twin see this one\nEvent ID: ${"9".repeat(64)}\nFrom: Mallory (hex: ${"8".repeat(64)})\n` +
          `Parsed: parent=${"7".repeat(64)}, root=${"6".repeat(64)}\nSeparate question`,
      );
      expect(quoted).toContain("Mallory");
      expect(parseSteeredMessage(quoted)).toMatchObject({
        eventId: STEERED_ID,
        senderPubkey: OWNER,
        replyToId: ROOT_ID,
        threadRootId: ROOT_ID,
      });
      expect(parseSteeredMessage(quoted).senderName).toBeUndefined();
    });

    it("starts a thread for a message that has no root, and reads a named channel", () => {
      const topLevel = STEERED.replace(/^Parsed:.*$/m, "").replace(
        `Channel: ${CHANNEL}`,
        `Channel: support (#${CHANNEL})`,
      );
      const meta = parseSteeredMessage(topLevel);
      expect(meta.threadRootId).toBe(STEERED_ID);
      expect(meta.replyToId).toBeUndefined();
      expect(meta.channelId).toBe(CHANNEL);
    });

    it("takes no id that is not 64 hex", () => {
      const meta = parseSteeredMessage(STEERED.replace(STEERED_ID, "0a03"));
      expect(meta).not.toHaveProperty("eventId");
      expect(meta.senderPubkey).toBe(OWNER);
    });
  });

  describe("extractPostedEventId", () => {
    const line = `posted (42 characters) · event_id ${POSTED_ID}`;

    it("takes the 64 hex characters after `event_id ` from a string or from content blocks", () => {
      expect(extractPostedEventId(line)).toBe(POSTED_ID);
      expect(extractPostedEventId([{ type: "text", text: line }])).toBe(POSTED_ID);
      expect(extractPostedEventId({ content: [{ type: "text", text: line }] })).toBe(POSTED_ID);
    });

    it("takes nothing else of the result", () => {
      for (const output of [
        "posted (42 characters)",
        `event_id ${POSTED_ID.slice(0, 63)}`,
        `event_id ${POSTED_ID}0`,
        `event_id ${POSTED_ID.toUpperCase()}`,
        `event-id ${POSTED_ID}`,
        `the id ${POSTED_ID} was posted`,
        `xevent_id ${POSTED_ID}`,
        [{ type: "image", data: line }],
        { text: 7 },
        null,
        undefined,
        42,
      ]) {
        expect(extractPostedEventId(output), JSON.stringify(output)).toBeUndefined();
      }
    });

    it("looks only at the start of a long result", () => {
      expect(extractPostedEventId("x".repeat(5000) + ` event_id ${POSTED_ID}`)).toBeUndefined();
    });
  });

  describe("the reply tool's span", () => {
    const runTool = (toolName: string, output: unknown, isError = false) => {
      exporter.reset();
      const handle = startTurn();
      handle.startToolSpan("toolu_1", toolName, { reply_to: OPENING_ID, text: "answer" });
      handle.endToolSpan("toolu_1", output, isError);
      handle.end("OK", "done");
      return only(`tool:${toolName}`);
    };

    it("carries the id of the message it posted", () => {
      const span = runTool("mcp__buzz-thread__reply", [
        { type: "text", text: `posted (6 characters) · event_id ${POSTED_ID}` },
      ]);
      expect(span.attributes["nostr.posted_event_id"]).toBe(POSTED_ID);
      // Still metadata only: the result's text is not in the span, nor is it anything but a size
      expect(span.attributes["tool.output_bytes"]).toBeGreaterThan(0);
      expect(JSON.stringify(span.attributes)).not.toContain("posted (6 characters)");
      expect(span.attributes["output.value"]).toBeUndefined();
    });

    it("carries nothing when the old text has no id, or the call failed", () => {
      expect(
        runTool("mcp__buzz-thread__reply", "posted (6 characters)").attributes[
          "nostr.posted_event_id"
        ],
      ).toBeUndefined();
      expect(
        runTool(
          "mcp__buzz-thread__reply",
          `buzz could not post the reply: event_id ${POSTED_ID}`,
          true,
        ).attributes["nostr.posted_event_id"],
      ).toBeUndefined();
    });

    it("is for the posting tool only", () => {
      for (const toolName of ["mcp__buzz-thread__thread_read", "Bash", "mcp__other__reply"]) {
        expect(
          runTool(toolName, `event_id ${POSTED_ID}`).attributes["nostr.posted_event_id"],
          toolName,
        ).toBeUndefined();
      }
    });
  });

  describe("the steer span", () => {
    it("is a child of the turn's span, in its trace, with the steered message's details", () => {
      const handle = startTurn();
      const arrivedAt = Date.now() - 1500;
      handle.startSteerSpan({
        uuid: "uuid-1",
        text: STEERED,
        priority: "now",
        nostrMeta: parseSteeredMessage(STEERED),
        arrivedAt,
      });
      handle.endSteerSpan("uuid-1", "echo");
      handle.end("OK", "answered");

      const root = only("agent_turn");
      const span = only("steer");
      expect(span.spanContext().traceId).toBe(root.spanContext().traceId);
      expect(span.parentSpanContext?.spanId).toBe(root.spanContext().spanId);
      expect(span.attributes).toMatchObject({
        "openinference.span.kind": "CHAIN",
        "input.mime_type": "text/plain",
        "input.value": STEERED,
        "nostr.event_id": STEERED_ID,
        "nostr.sender_pubkey": OWNER,
        "nostr.reply_to_id": ROOT_ID,
        "nostr.thread_root_id": ROOT_ID,
        "nostr.channel_id": CHANNEL,
        "steer.priority": "now",
        "steer.delivered": true,
        "steer.end_reason": "echo",
        "steer.uuid": "uuid-1",
        // The turn's own session, as its other spans have it
        "session.id": ROOT_ID,
      });
      expect(span.attributes["nostr.sender_name"]).toBeUndefined();
      expect(span.status.code).toBe(1);
      // The span starts where the steer arrived, not when it was recorded
      expect(ms(span.startTime)).toBeGreaterThanOrEqual(arrivedAt - 1);
      expect(ms(span.startTime)).toBeLessThan(arrivedAt + 500);
      expect(ms(span.endTime)).toBeGreaterThan(ms(span.startTime) + 1000);
    });

    it("carries the sender's name when the block has one, and `later` when it was delivered later", () => {
      const handle = startTurn();
      const named = STEERED.replace(
        /From: npub1\S+ \(hex:/,
        "From: Bob Example (npub: npub1abc, hex:",
      );
      steer(handle, "uuid-1", named, "later");
      handle.end("OK", "answered");
      const span = only("steer");
      expect(span.attributes["nostr.sender_name"]).toBe("Bob Example");
      expect(span.attributes["steer.priority"]).toBe("later");
    });

    it("holds the message as the turn's prompt is held: secrets redacted, at most 8 KB", () => {
      const handle = startTurn();
      const secret = "nsec1vl029mgpspedva04g90vltkh6fvh240eqtv9xxl2xme3sqmxugjqwfuxff";
      steer(handle, "uuid-1", `${STEERED}\nmy key is ${secret}`);
      steer(handle, "uuid-2", `${STEERED}\n${"y".repeat(20000)}`);
      handle.end("OK", "answered");
      const [first, second] = finished("steer").sort((a, b) =>
        String(a.attributes["steer.uuid"]).localeCompare(String(b.attributes["steer.uuid"])),
      );
      expect(String(first.attributes["input.value"])).not.toContain(secret);
      expect(String(first.attributes["input.value"])).toContain("[REDACTED]");
      // The span's own attribute limit (8192 characters) cuts what truncatePayload left, as it does a turn's prompt
      const cut = String(second.attributes["input.value"]);
      expect(cut).toHaveLength(8192);
      expect(cut.startsWith("[New message")).toBe(true);
    });

    it("takes a message that has no Buzz event: the text, no ids", () => {
      const handle = startTurn();
      steer(handle, "uuid-1", "also check the totals");
      handle.endSteerSpan("uuid-1", "result");
      handle.end("OK", "answered");
      const span = only("steer");
      expect(span.attributes["input.value"]).toBe("also check the totals");
      expect(span.attributes["nostr.event_id"]).toBeUndefined();
      expect(span.attributes["steer.end_reason"]).toBe("result");
    });

    it("is one span for each steered message, each ended on its own", () => {
      const handle = startTurn();
      steer(handle, "uuid-1", STEERED.replace(STEERED_ID, "1".repeat(64)));
      steer(handle, "uuid-2", STEERED.replace(STEERED_ID, "2".repeat(64)));
      steer(handle, "uuid-3", STEERED.replace(STEERED_ID, "3".repeat(64)));
      handle.endSteerSpan("uuid-2", "echo");
      expect(finished("steer").map((span) => span.attributes["nostr.event_id"])).toEqual([
        "2".repeat(64),
      ]);
      handle.endSteerSpan("uuid-1", "result");
      handle.end("OK", "answered");
      const bySteer = Object.fromEntries(
        finished("steer").map((span) => [
          span.attributes["nostr.event_id"],
          span.attributes["steer.delivered"],
        ]),
      );
      expect(bySteer).toEqual({
        ["1".repeat(64)]: true,
        ["2".repeat(64)]: true,
        ["3".repeat(64)]: false,
      });
    });

    it.each(["OK", "ERROR", "CANCELLED"] as const)(
      "closes a message the SDK never took when the turn ends %s: not delivered, and before the turn's span",
      (status) => {
        const handle = startTurn();
        steer(handle, "uuid-1");
        handle.end(status, status === "OK" ? "answered" : "stopped");
        const root = only("agent_turn");
        const span = only("steer");
        expect(span.attributes["steer.delivered"]).toBe(false);
        expect(span.attributes["steer.end_reason"]).toBe("turn_end");
        // Not an error of the turn
        expect(span.status.code).toBe(0);
        // Ended first: the order the spans reached the exporter (the SDK's timestamps can differ by under a millisecond)
        const order = exporter.getFinishedSpans();
        expect(order.indexOf(span)).toBeLessThan(order.indexOf(root));
        expect(span.parentSpanContext?.spanId).toBe(root.spanContext().spanId);
      },
    );

    it("never leaves a span open: after the turn ended nothing starts or ends", () => {
      const handle = startTurn();
      steer(handle, "uuid-1");
      handle.end("OK", "answered");
      const count = exporter.getFinishedSpans().length;
      steer(handle, "uuid-late");
      handle.endSteerSpan("uuid-late", "echo");
      handle.endSteerSpan("uuid-1", "echo");
      handle.end("ERROR", "twice");
      expect(exporter.getFinishedSpans()).toHaveLength(count);
      expect(only("steer").attributes["steer.delivered"]).toBe(false);
    });

    it("ignores a uuid it has not seen, a second end of one, and a second start of one", () => {
      const handle = startTurn();
      handle.endSteerSpan("nobody", "echo");
      steer(handle, "uuid-1");
      steer(handle, "uuid-1", `${STEERED}\nagain`);
      handle.endSteerSpan("uuid-1", "echo");
      handle.endSteerSpan("uuid-1", "result");
      handle.end("OK", "answered");
      const span = only("steer");
      expect(span.attributes["steer.end_reason"]).toBe("echo");
      expect(String(span.attributes["input.value"])).not.toContain("again");
    });

    it("stops at a cap, so that a flood of mentions cannot flood the trace, and says so on the turn", () => {
      const handle = startTurn();
      for (let i = 0; i < MAX_STEER_SPANS_PER_TURN + 25; i++) steer(handle, `uuid-${i}`);
      handle.end("OK", "answered");
      expect(finished("steer")).toHaveLength(MAX_STEER_SPANS_PER_TURN);
      const capped = only("agent_turn").events.filter(
        (event) => event.name === "steer_spans_capped",
      );
      expect(capped).toHaveLength(1);
      expect(capped[0].attributes).toMatchObject({ "steer.span_cap": MAX_STEER_SPANS_PER_TURN });
    });

    it("keeps the turn's other spans as they were", () => {
      const handle = startTurn();
      handle.startToolSpan("toolu_1", "Read", { file_path: "/tmp/a" });
      steer(handle, "uuid-1");
      handle.endToolSpan("toolu_1", "contents", false);
      handle.endSteerSpan("uuid-1", "echo");
      handle.end("OK", "answered");
      expect(
        exporter
          .getFinishedSpans()
          .map((span) => span.name)
          .sort(),
      ).toEqual(["agent_turn", "llm:sonnet", "steer", "tool:Read"]);
    });
  });

  it("starts no steer span when tracing was never started: the handle is the only way in", () => {
    // Without an OTLP endpoint there is no service, hence no handle for the agent to call.
    delete process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT;
    delete process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
    expect(initTelemetry()).toBeNull();
  });
});
