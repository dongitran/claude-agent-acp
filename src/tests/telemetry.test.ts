import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  initTelemetry,
  parseNostrContext,
  redactSecrets,
  truncatePayload,
  computeOutputBytes,
  RobustDifferentialCostTracker,
  _resetProcessSecretsCache,
} from "../telemetry.js";

describe("Telemetry Unit Test Suite", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
    _resetProcessSecretsCache();
  });

  afterEach(() => {
    process.env = originalEnv;
    _resetProcessSecretsCache();
    vi.restoreAllMocks();
  });

  describe("Fail-Open Initialization & Environment Guards", () => {
    it("returns null when neither OTEL_EXPORTER_OTLP_TRACES_ENDPOINT nor OTEL_EXPORTER_OTLP_ENDPOINT is configured", () => {
      delete process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT;
      delete process.env.OTEL_EXPORTER_OTLP_ENDPOINT;

      const service = initTelemetry();
      expect(service).toBeNull();
    });

    it("initializes TelemetryService when traces endpoint is set", async () => {
      process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = "http://127.0.0.1:4318/v1/traces";
      process.env.OTEL_SERVICE_NAME = "claude-twin";
      process.env.PHOENIX_PROJECT_NAME = "test-project";

      const service = initTelemetry();
      expect(service).not.toBeNull();
      expect(typeof service?.startTurnSpan).toBe("function");
      expect(typeof service?.backgroundFlush).toBe("function");
      expect(typeof service?.shutdown).toBe("function");

      await service?.shutdown();
    });
  });

  describe("Secret Sanitizer & Redaction", () => {
    it("redacts Bech32 Nostr private keys (nsec1...) in lowercase and uppercase", () => {
      const lowerNsec = "nsec1vl029mgpspedva04g90vltkh6fvh240eqtv9xxl2xme3sqmxugjqwfuxff";
      const upperNsec = "NSEC1VL029MGPSPEDVA04G90VLTKH6FVH240EQTV9XXL2XME3SQMXUGJQWFUxFF";

      const input = `My secret key is ${lowerNsec} and backup is ${upperNsec}.`;
      const sanitized = redactSecrets(input);

      expect(sanitized).not.toContain(lowerNsec);
      expect(sanitized).not.toContain(upperNsec);
      expect(sanitized).toContain("[REDACTED]");
    });

    it("redacts sensitive environment process secrets when configured", () => {
      process.env.ANTHROPIC_API_KEY = "sk-ant-api03-abcdef1234567890-test-key";
      process.env.BUZZ_PRIVATE_KEY = "buzz-priv-secret-key-12345";
      _resetProcessSecretsCache();

      const input =
        "Connecting with sk-ant-api03-abcdef1234567890-test-key and buzz-priv-secret-key-12345";
      const sanitized = redactSecrets(input);

      expect(sanitized).not.toContain("buzz-priv-secret-key-12345");
      expect(sanitized).not.toContain("sk-ant-api03-abcdef1234567890-test-key");
      expect(sanitized).toContain("[REDACTED]");
    });

    it("truncates payloads longer than maxLen with [TRUNCATED] suffix", () => {
      const longText = "a".repeat(10000);
      const truncated = truncatePayload(longText, 100);

      expect(truncated.length).toBe(100 + "... [TRUNCATED]".length);
      expect(truncated.endsWith("... [TRUNCATED]")).toBe(true);
    });

    it("preserves short payloads without truncation", () => {
      const short = "Hello world";
      expect(truncatePayload(short, 100)).toBe("Hello world");
    });
  });

  describe("computeOutputBytes Precision & Safe Serialization", () => {
    it("returns 0 for undefined or null", () => {
      expect(computeOutputBytes(undefined)).toBe(0);
      expect(computeOutputBytes(null)).toBe(0);
    });

    it("returns exact byte length for ASCII and UTF-8 strings", () => {
      expect(computeOutputBytes("abc")).toBe(3);
      expect(computeOutputBytes("Xin chào")).toBe(Buffer.byteLength("Xin chào", "utf8"));
    });

    it("returns buffer length for Buffer objects", () => {
      const buf = Buffer.from([1, 2, 3, 4, 5]);
      expect(computeOutputBytes(buf)).toBe(5);
    });

    it("correctly handles plain objects without throwing", () => {
      const obj = { success: true, count: 42, name: "test" };
      const expected = Buffer.byteLength(JSON.stringify(obj), "utf8");
      expect(computeOutputBytes(obj)).toBe(expected);
    });

    it("correctly handles arrays without throwing", () => {
      const arr = [1, 2, "three", { four: 4 }];
      const expected = Buffer.byteLength(JSON.stringify(arr), "utf8");
      expect(computeOutputBytes(arr)).toBe(expected);
    });

    it("handles circular objects gracefully by falling back to String(val)", () => {
      const circular: Record<string, unknown> = { a: 1 };
      circular.self = circular;

      expect(() => computeOutputBytes(circular)).not.toThrow();
      expect(computeOutputBytes(circular)).toBeGreaterThan(0);
    });
  });

  describe("RobustDifferentialCostTracker", () => {
    it("computes incremental differential turn costs correctly", () => {
      const tracker = new RobustDifferentialCostTracker();
      const sessionId = "sess-cost-1";

      // First cumulative reading: $0.050000
      const cost1 = tracker.computeTurnCost(sessionId, 0.05);
      expect(cost1).toBe(0.05);

      // Second cumulative reading: $0.120000 -> delta = $0.070000
      const cost2 = tracker.computeTurnCost(sessionId, 0.12);
      expect(cost2).toBe(0.07);

      // Third cumulative reading identical: $0.120000 -> delta = $0
      const cost3 = tracker.computeTurnCost(sessionId, 0.12);
      expect(cost3).toBe(0);

      // Backend resets session (cumulative drops to $0.030000) -> delta = $0.030000
      const costAfterReset = tracker.computeTurnCost(sessionId, 0.03);
      expect(costAfterReset).toBe(0.03);

      // Purge session
      tracker.purgeSession(sessionId);
      const costAfterPurge = tracker.computeTurnCost(sessionId, 0.02);
      expect(costAfterPurge).toBe(0.02);
    });

    it("handles invalid or non-numeric cost values safely", () => {
      const tracker = new RobustDifferentialCostTracker();
      expect(tracker.computeTurnCost("sess-inv", undefined)).toBe(0);
      expect(tracker.computeTurnCost("sess-inv", NaN)).toBe(0);
      expect(tracker.computeTurnCost("sess-inv", -1)).toBe(0);
    });
  });

  describe("Nostr Context Envelope Parser", () => {
    it("extracts channel, thread, event, and sender details from standard Nostr envelope", () => {
      const prompt = `Channel: general (#e4d63428-15aa-49be-a006-2182b8efca92)
Thread root: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
Event ID: fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210
parent=1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff
From: Alice (npub: npub1author888, hex: 222233334444555566667777888899990000aaaabbbbccccddddeeeeffff1111)

What is the latest status?`;

      const meta = parseNostrContext(prompt);
      expect(meta.channelId).toBe("e4d63428-15aa-49be-a006-2182b8efca92");
      expect(meta.channelName).toBe("general");
      expect(meta.eventId).toBe("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210");
      expect(meta.threadRootId).toBe(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      );
      expect(meta.replyToId).toBe(
        "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
      );
      expect(meta.senderPubkey).toBe(
        "222233334444555566667777888899990000aaaabbbbccccddddeeeeffff1111",
      );
      expect(meta.senderName).toBe("Alice");
    });

    it("isolates the active envelope from steered batch envelopes using lastIndexOf", () => {
      const steeredPrompt = `Channel: oldchannel (#11111111-2222-3333-4444-555555555555)
Event ID: 1111111111111111111111111111111111111111111111111111111111111111
From: Bob (hex: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)

First message

Channel: latestchannel (#99999999-8888-7777-6666-555555555555)
Event ID: 9999999999999999999999999999999999999999999999999999999999999999
Thread root: 0000000000000000000000000000000000000000000000000000000000000000
From: Charlie (hex: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc)

Second steered message`;

      const meta = parseNostrContext(steeredPrompt);
      expect(meta.channelId).toBe("99999999-8888-7777-6666-555555555555");
      expect(meta.channelName).toBe("latestchannel");
      expect(meta.eventId).toBe("9999999999999999999999999999999999999999999999999999999999999999");
      expect(meta.threadRootId).toBe(
        "0000000000000000000000000000000000000000000000000000000000000000",
      );
      expect(meta.senderName).toBe("Charlie");
    });

    it("handles DM channel UUIDs and Unicode sender display names", () => {
      const dmPrompt = `Channel: e4d63428-15aa-49be-a006-2182b8efca92
Event ID: 3333333333333333333333333333333333333333333333333333333333333333
From: Trần Quốc Tuấn 🚀 (hex: 4444444444444444444444444444444444444444444444444444444444444444)

Xin chào bạn`;

      const meta = parseNostrContext(dmPrompt);
      expect(meta.channelId).toBe("e4d63428-15aa-49be-a006-2182b8efca92");
      expect(meta.senderName).toBe("Trần Quốc Tuấn 🚀");
      expect(meta.senderPubkey).toBe(
        "4444444444444444444444444444444444444444444444444444444444444444",
      );
    });

    it("extracts Nostr envelope from multi-block prompt batches", () => {
      const blocks = [
        "[Base system instructions that are 15KB long...]",
        "[Agent specific instructions...]",
        "[Agent Memory — core]",
        "[Context: previous events...]",
        `Channel: twin (#01fd5ea1-0089-471b-9aac-96e54993a4ab)
Thread root: 6d6154848b8097cfa1ba068e7bd0d8a300000000000000000000000000000000
Event ID: 22bf0d552a1a6e2dc95987f3979abac8142e06e9e6f11c9002b418fdca207c33
From: elio.tran (npub: npub1author, hex: a666af900255fef167708bc019a1b56c0c801416b86deebe3a4e1efda1fe5b0c)

@Claude Twin Hello from test`,
      ];

      // Combining text blocks
      const combined = blocks.join("\n\n");
      const meta = parseNostrContext(combined);

      expect(meta.channelId).toBe("01fd5ea1-0089-471b-9aac-96e54993a4ab");
      expect(meta.channelName).toBe("twin");
      expect(meta.eventId).toBe("22bf0d552a1a6e2dc95987f3979abac8142e06e9e6f11c9002b418fdca207c33");
      expect(meta.threadRootId).toBe(
        "6d6154848b8097cfa1ba068e7bd0d8a300000000000000000000000000000000",
      );
      expect(meta.senderName).toBe("elio.tran");
      expect(meta.senderPubkey).toBe(
        "a666af900255fef167708bc019a1b56c0c801416b86deebe3a4e1efda1fe5b0c",
      );
    });
  });

  describe("Turn Span Lifecycle & Single-Flight Coalesced Flush", () => {
    it("manages turn span and tool spans cleanly", async () => {
      process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = "http://127.0.0.1:4318/v1/traces";
      const service = await initTelemetry();
      expect(service).not.toBeNull();

      const handle = service!.startTurnSpan(
        "sess-123",
        "Hello test",
        "claude-3-5-sonnet-20241022",
        {
          channelId: "chan-1",
          senderPubkey: "222233334444555566667777888899990000aaaabbbbccccddddeeeeffff1111",
          threadRootId: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        },
      );

      // Start tool span
      handle.startToolSpan("call-1", "bash", { command: "ls -la" });

      // End tool span
      handle.endToolSpan("call-1", "total 0\n-rw-r--r-- file.txt", false, "bash");

      // Record usage
      handle.recordUsage({
        input_tokens: 150,
        output_tokens: 50,
      });

      // End turn span
      handle.end("OK", "Directory listed successfully");

      // Flush should coalesce without throwing
      service!.backgroundFlush(100);
      service!.backgroundFlush(100); // duplicate call coalesced

      await service!.shutdown();
    });

    it("sweeps active tool spans if turn ends prematurely", async () => {
      process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT = "http://127.0.0.1:4318/v1/traces";
      const service = await initTelemetry();
      expect(service).not.toBeNull();

      const handle = service!.startTurnSpan("sess-orphan", "Cancel test");
      handle.startToolSpan("call-orphan", "read_file", { path: "/tmp/foo" });

      // End turn without explicitly closing call-orphan
      handle.end("CANCELLED", "Turn was interrupted");

      // Sweep should close the orphan tool span safely
      service!.backgroundFlush(100);
      await service!.shutdown();
    });
  });
});
