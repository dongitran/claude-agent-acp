import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { randomUUID } from "node:crypto";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  type ReadableSpan,
} from "@opentelemetry/sdk-trace-node";
import { ClaudeAcpAgent, type AcpClient } from "../acp-agent.js";
import { initTelemetry, type TelemetryService, type TurnSpanHandle } from "../telemetry.js";
import { Pushable } from "../utils.js";
import {
  mockSessionState,
  successfulResultMessage,
  userEcho,
  wrapQuery,
} from "./session-doubles.js";

vi.mock("@anthropic-ai/claude-agent-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@anthropic-ai/claude-agent-sdk")>();
  const refused = () => {
    throw new Error("Unexpected real SDK call");
  };
  return {
    ...actual,
    query: vi.fn(refused),
    getSessionMessages: vi.fn(async () => []),
    getSubagentMessages: vi.fn(async () => []),
    listSessions: vi.fn(async () => []),
    getSessionInfo: vi.fn(async () => undefined),
    deleteSession: vi.fn(refused),
    forkSession: vi.fn(refused),
    importSessionToStore: vi.fn(refused),
  };
});

const OWNER = "0a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9";
const AGENT = "1f2e3d4c5b6a79881706f5e4d3c2b1a01f2e3d4c5b6a79881706f5e4d3c2b1a0";
const CHANNEL = "0b5c1d2e-3f40-4a5b-8c6d-7e8f90a1b2c3";
const ROOT_ID = "0000000000000000000000000000000000000000000000000000000000000a01";
const OPENING_ID = "0000000000000000000000000000000000000000000000000000000000000a02";

/** The two blocks buzz-acp sends to open a turn: its context, then the event. */
const OPENING_BLOCKS = [
  {
    type: "text" as const,
    text: `[Context]\nChannel: support (#${CHANNEL})\nThread root: ${ROOT_ID}`,
  },
  {
    type: "text" as const,
    text: `[Buzz event: @mention]
Event ID: ${OPENING_ID}
Channel: support (#${CHANNEL})
Kind: 9
From: Alice Example (npub: npub1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq, hex: ${OWNER})
Time: 2026-11-08T04:20:00+00:00
Content: @Twin first question
Parsed: root=${ROOT_ID}, mentions=[${AGENT}]`,
  },
];

/** What buzz-acp sends through `_session/steering`: one block, one event, no display name in `From:`. */
function steeredBlock(eventId: string, question: string): string {
  return `[New message — arrived while you were working]

[Buzz event: @mention]
Event ID: ${eventId}
Channel: ${CHANNEL}
Kind: 9
From: npub1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq (hex: ${OWNER})
Time: 2026-11-08T04:20:04+00:00
Content: @Twin ${question}
Tags: [["h","${CHANNEL}"],["e","${ROOT_ID}","","reply"],["p","${AGENT}"]]
Parsed: parent=${ROOT_ID}, root=${ROOT_ID}, mentions=[${AGENT}]

Note: A new message arrived while you were working. Continue your in-progress work and incorporate the new message if it's relevant; if it's unrelated, you may briefly acknowledge it and carry on.`;
}

const STEER_1 = { id: "1".repeat(64), question: "Which SAP table holds material descriptions?" };
const STEER_2 = { id: "2".repeat(64), question: "What is table CKMLPP used for?" };
const steerRequest = (steer: { id: string; question: string }) => ({
  sessionId: "test-session",
  prompt: [{ type: "text" as const, text: steeredBlock(steer.id, steer.question) }],
});

const idle = { type: "system", subtype: "session_state_changed", state: "idle" };

/** The `result` of the cycle a priority:'now' steer aborted (`origin` human, stopped on a tool call). */
function interruptedCycleResult() {
  return successfulResultMessage({
    stop_reason: "tool_use",
    origin: { kind: "human" },
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  });
}

/** The `result` that names the messages it answered, as the CLI stamps it. */
const stampedResult = (...uuids: string[]) =>
  successfulResultMessage({
    origin: { kind: "human" },
    user_message_uuid: uuids[uuids.length - 1],
    user_message_uuids: uuids,
  });

const waitFor = async (cond: () => boolean) => {
  for (let i = 0; i < 400; i++) {
    if (cond()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("waitFor timed out");
};

describe("steered messages in the trace", () => {
  const originalEnv = process.env;
  let exporter: InMemorySpanExporter;
  let service: TelemetryService;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    // Only around initTelemetry: an agent built while the variable is set would start an OTLP exporter of its own.
    process.env = {
      ...originalEnv,
      OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: "http://127.0.0.1:4318/v1/traces",
    };
    service = initTelemetry(undefined, { spanProcessors: [new SimpleSpanProcessor(exporter)] })!;
    process.env = { ...originalEnv };
    delete process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT;
    delete process.env.OTEL_EXPORTER_OTLP_ENDPOINT;
  });

  afterEach(async () => {
    await service.shutdown();
    process.env = originalEnv;
  });

  function createAgent(telemetry: TelemetryService | null = service) {
    const errors: unknown[][] = [];
    const agent = new ClaudeAcpAgent({ sessionUpdate: async () => {} } as unknown as AcpClient, {
      log: () => {},
      error: (...args: unknown[]) => errors.push(args),
    });
    agent.telemetry = telemetry;
    return { agent, errors };
  }

  function inject(
    agent: ClaudeAcpAgent,
    makeGenerator: (input: Pushable<any>) => AsyncGenerator<any>,
    overrides: Record<string, unknown> = {},
  ) {
    const input = new Pushable<any>();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(makeGenerator(input)),
      input,
      ...overrides,
    });
    return input;
  }

  const finished = (name: string): ReadableSpan[] =>
    exporter.getFinishedSpans().filter((span) => span.name === name);
  const only = (name: string): ReadableSpan => {
    expect(finished(name), `spans named ${name}`).toHaveLength(1);
    return finished(name)[0];
  };
  /** Whether `first` ended before `second`: the order the spans reached the exporter. Not their timestamps, which the
   *  SDK anchors on each span's own start and so can differ by under a millisecond. */
  const endedBefore = (first: ReadableSpan, second: ReadableSpan) => {
    const order = exporter.getFinishedSpans();
    return order.indexOf(first) < order.indexOf(second);
  };
  const steerSpans = () =>
    Object.fromEntries(
      finished("steer").map((span) => [span.attributes["nostr.event_id"] as string, span]),
    );

  const startTurn = (agent: ClaudeAcpAgent) =>
    agent.prompt({ sessionId: "test-session", prompt: OPENING_BLOCKS });

  it("shows a steered message as a child of the turn it joined, and ends it when the SDK takes it", async () => {
    const { agent } = createAgent();
    const captured: any[] = [];
    const echo = Promise.withResolvers<void>();
    const finish = Promise.withResolvers<void>();
    inject(agent, (input) =>
      (async function* () {
        const iter = input[Symbol.asyncIterator]();
        const { value: opening } = await iter.next();
        yield userEcho(opening);
        const { value: steered } = await iter.next();
        captured.push(steered);
        await echo.promise;
        yield userEcho(steered);
        await finish.promise;
        yield successfulResultMessage();
        yield idle;
      })(),
    );

    const turn = startTurn(agent);
    await waitFor(() => !!agent.sessions["test-session"]?.activeTurn);
    await expect(agent.steer(steerRequest(STEER_1))).resolves.toEqual({ outcome: "injected" });

    // Open until the SDK takes the message: nothing of it is exported yet
    expect(finished("steer")).toHaveLength(0);

    echo.resolve();
    await waitFor(() => finished("steer").length === 1);
    const span = only("steer");
    expect(span.attributes).toMatchObject({
      "openinference.span.kind": "CHAIN",
      "nostr.event_id": STEER_1.id,
      "nostr.sender_pubkey": OWNER,
      "nostr.reply_to_id": ROOT_ID,
      "steer.priority": "now",
      "steer.delivered": true,
      "steer.end_reason": "echo",
      "steer.uuid": captured[0].uuid,
    });
    expect(String(span.attributes["input.value"])).toContain(STEER_1.question);
    expect(span.attributes["nostr.sender_name"]).toBeUndefined();
    // The turn is still running: its own span is open
    expect(finished("agent_turn")).toHaveLength(0);

    finish.resolve();
    await expect(turn).resolves.toMatchObject({ stopReason: "end_turn" });
    const root = only("agent_turn");
    expect(span.parentSpanContext?.spanId).toBe(root.spanContext().spanId);
    expect(span.spanContext().traceId).toBe(root.spanContext().traceId);
    expect(endedBefore(span, root)).toBe(true);
    // The turn's own span is the opening message's, as before
    expect(root.attributes["nostr.event_id"]).toBe(OPENING_ID);
    expect(finished("steer")).toHaveLength(1);

    // And the steer itself was delivered as it always was
    expect(captured).toHaveLength(1);
    expect(captured[0].priority).toBe("now");
    expect(JSON.stringify(captured[0].message.content)).toContain(STEER_1.question);
  });

  it("records one span for each steered message, each with its own ids and text", async () => {
    const { agent } = createAgent();
    const finish = Promise.withResolvers<void>();
    inject(agent, (input) =>
      (async function* () {
        const iter = input[Symbol.asyncIterator]();
        const { value: opening } = await iter.next();
        yield userEcho(opening);
        const { value: first } = await iter.next();
        const { value: second } = await iter.next();
        // The echoes come in the other order: each span ends on its own message's
        yield userEcho(second);
        yield userEcho(first);
        await finish.promise;
        yield successfulResultMessage();
        yield idle;
      })(),
    );

    const turn = startTurn(agent);
    await waitFor(() => !!agent.sessions["test-session"]?.activeTurn);
    await agent.steer(steerRequest(STEER_1));
    await agent.steer(steerRequest(STEER_2));
    await waitFor(() => finished("steer").length === 2);
    finish.resolve();
    await turn;

    const spans = steerSpans();
    expect(Object.keys(spans).sort()).toEqual([STEER_1.id, STEER_2.id]);
    expect(String(spans[STEER_1.id].attributes["input.value"])).toContain(STEER_1.question);
    expect(String(spans[STEER_2.id].attributes["input.value"])).toContain(STEER_2.question);
    expect(spans[STEER_1.id].attributes["steer.delivered"]).toBe(true);
    expect(spans[STEER_2.id].attributes["steer.delivered"]).toBe(true);
    const parents = new Set(finished("steer").map((span) => span.parentSpanContext?.spanId));
    expect(parents).toEqual(new Set([only("agent_turn").spanContext().spanId]));
    expect(finished("agent_turn")).toHaveLength(1);
  });

  it("ends a steer at the result that names it when its echo never came", async () => {
    const { agent } = createAgent();
    const finish = Promise.withResolvers<void>();
    inject(agent, (input) =>
      (async function* () {
        const iter = input[Symbol.asyncIterator]();
        const { value: opening } = await iter.next();
        yield userEcho(opening);
        const { value: steered } = await iter.next();
        yield { ...interruptedCycleResult(), user_message_uuid: opening.uuid };
        // No echo for the steered message: the result that answers it names it
        yield stampedResult(steered.uuid);
        await waitFor(() => finished("steer").length === 1);
        await finish.promise;
        yield idle;
      })(),
    );

    const turn = startTurn(agent);
    await waitFor(() => !!agent.sessions["test-session"]?.activeTurn);
    await agent.steer(steerRequest(STEER_1));
    await waitFor(() => finished("steer").length === 1);
    expect(only("steer").attributes).toMatchObject({
      "steer.delivered": true,
      "steer.end_reason": "result",
    });
    finish.resolve();
    await expect(turn).resolves.toMatchObject({ stopReason: "end_turn" });
    expect(only("agent_turn").spanContext().traceId).toBe(only("steer").spanContext().traceId);
  });

  describe("a message the SDK never took before the turn ended", () => {
    /** A turn working on its answer; every message pushed after the opening one is read, never echoed. */
    function workingTurn(agent: ClaudeAcpAgent, steersToRead: number) {
      const afterCancel = Promise.withResolvers<void>();
      const echoFirst = Promise.withResolvers<void>();
      inject(agent, (input) =>
        (async function* () {
          const iter = input[Symbol.asyncIterator]();
          const { value: opening } = await iter.next();
          yield userEcho(opening);
          const steers: any[] = [];
          for (let i = 0; i < steersToRead; i++) steers.push((await iter.next()).value);
          yield interruptedCycleResult();
          if (steersToRead > 1) {
            await echoFirst.promise;
            yield userEcho(steers[0]);
          }
          await afterCancel.promise;
          yield idle; // the interrupt's trailer: settles the turn "cancelled"
        })(),
      );
      return { afterCancel, echoFirst };
    }

    it("closes its span with the turn, not delivered, when the turn is cancelled", async () => {
      const { agent } = createAgent();
      const { afterCancel } = workingTurn(agent, 1);
      const turn = startTurn(agent);
      await waitFor(() => !!agent.sessions["test-session"]?.activeTurn);
      await agent.steer(steerRequest(STEER_1));
      await waitFor(() => agent.sessions["test-session"]?.activeTurn?.steeredSettle !== undefined);
      expect(finished("steer")).toHaveLength(0);

      await agent.cancel({ sessionId: "test-session" });
      afterCancel.resolve();
      await expect(turn).resolves.toMatchObject({ stopReason: "cancelled" });

      const span = only("steer");
      expect(span.attributes).toMatchObject({
        "steer.delivered": false,
        "steer.end_reason": "turn_end",
        "nostr.event_id": STEER_1.id,
      });
      const root = only("agent_turn");
      expect(root.attributes["claude.stop_reason"]).toBe("cancelled");
      expect(span.parentSpanContext?.spanId).toBe(root.spanContext().spanId);
      expect(endedBefore(span, root)).toBe(true);
    });

    it("leaves the message the SDK took delivered and the one it did not, not, in one turn", async () => {
      const { agent } = createAgent();
      const { afterCancel, echoFirst } = workingTurn(agent, 2);
      const turn = startTurn(agent);
      await waitFor(() => !!agent.sessions["test-session"]?.activeTurn);
      await agent.steer(steerRequest(STEER_1));
      await agent.steer(steerRequest(STEER_2));
      echoFirst.resolve();
      await waitFor(() => finished("steer").length === 1);

      await agent.cancel({ sessionId: "test-session" });
      afterCancel.resolve();
      await expect(turn).resolves.toMatchObject({ stopReason: "cancelled" });

      const spans = steerSpans();
      expect(spans[STEER_1.id].attributes["steer.delivered"]).toBe(true);
      expect(spans[STEER_2.id].attributes["steer.delivered"]).toBe(false);
      expect(finished("agent_turn")).toHaveLength(1);
    });

    it("ignores an echo that comes after the turn is over", async () => {
      const { agent } = createAgent();
      const afterCancel = Promise.withResolvers<void>();
      const late = Promise.withResolvers<void>();
      const consumed = Promise.withResolvers<void>();
      inject(agent, (input) =>
        (async function* () {
          const iter = input[Symbol.asyncIterator]();
          const { value: opening } = await iter.next();
          yield userEcho(opening);
          const { value: steered } = await iter.next();
          yield interruptedCycleResult();
          await afterCancel.promise;
          yield idle; // the interrupt's trailer: settles the turn "cancelled"
          await late.promise;
          yield userEcho(steered);
          consumed.resolve();
        })(),
      );
      const turn = startTurn(agent);
      await waitFor(() => !!agent.sessions["test-session"]?.activeTurn);
      await agent.steer(steerRequest(STEER_1));
      await waitFor(() => agent.sessions["test-session"]?.activeTurn?.steeredSettle !== undefined);
      await agent.cancel({ sessionId: "test-session" });
      afterCancel.resolve();
      await expect(turn).resolves.toMatchObject({ stopReason: "cancelled" });
      expect(only("steer").attributes["steer.delivered"]).toBe(false);

      late.resolve();
      await consumed.promise;
      expect(finished("steer")).toHaveLength(1);
      expect(only("steer").attributes["steer.delivered"]).toBe(false);
    });
  });

  it("marks a steer `later` while a permission or elicitation request is open", async () => {
    const { agent } = createAgent();
    const input = new Pushable<any>();
    const push = vi.spyOn(input, "push");
    const handle = service.startTurnSpan(
      "acp-session",
      "[Buzz event: @mention]\nEvent ID: " + OPENING_ID,
    );
    agent.sessions["test-session"] = mockSessionState({
      input,
      openUserInputs: new Set([{}]),
      turnQueue: [
        {
          promptUuid: "running",
          isLocalOnlyCommand: false,
          settled: false,
          resolve: () => {},
          reject: () => {},
          telemetryHandle: handle,
        },
      ],
    });

    await expect(agent.steer(steerRequest(STEER_1))).resolves.toEqual({ outcome: "injected" });
    expect(push.mock.calls[0][0].priority).toBe("later");
    handle.end("OK", "answered");
    expect(only("steer").attributes["steer.priority"]).toBe("later");
  });

  it("records no steer when no turn is running: the message opens a turn of its own", async () => {
    const { agent } = createAgent();
    inject(agent, (input) =>
      (async function* () {
        const iter = input[Symbol.asyncIterator]();
        const { value } = await iter.next();
        yield userEcho(value);
        yield successfulResultMessage();
        yield idle;
      })(),
    );

    await expect(agent.steer(steerRequest(STEER_1))).resolves.toEqual({
      outcome: "startedNewTurn",
    });
    await waitFor(() => finished("agent_turn").length === 1);
    expect(finished("steer")).toHaveLength(0);
    const root = only("agent_turn");
    expect(root.attributes["nostr.event_id"]).toBe(STEER_1.id);
    expect(String(root.attributes["input.value"])).toContain(STEER_1.question);
  });

  it("injects a steer the same way with tracing off, and with it on", async () => {
    const pushed: Record<"on" | "off", any> = { on: undefined, off: undefined };
    for (const mode of ["off", "on"] as const) {
      const { agent } = createAgent(mode === "on" ? service : null);
      const turn = agent.telemetry
        ? service.startTurnSpan("acp-session", OPENING_BLOCKS[1].text)
        : undefined;
      const input = new Pushable<any>();
      const push = vi.spyOn(input, "push");
      agent.sessions["test-session"] = mockSessionState({
        input,
        turnQueue: [
          {
            promptUuid: "running",
            isLocalOnlyCommand: false,
            settled: false,
            resolve: () => {},
            reject: () => {},
            ...(turn ? { telemetryHandle: turn } : {}),
          },
        ],
      });
      await expect(agent.steer(steerRequest(STEER_1))).resolves.toEqual({ outcome: "injected" });
      pushed[mode] = { ...push.mock.calls[0][0], uuid: "<uuid>" };
      const queued = agent.sessions["test-session"].turnQueue![0];
      expect(queued.steeredEchoes?.size).toBe(1);
      expect(queued.steeredUuids?.size).toBe(1);
      turn?.end("OK", "answered");
    }
    expect(pushed.on).toEqual(pushed.off);
    expect(finished("steer")).toHaveLength(1);
  });

  it("delivers the steer when recording it fails, and says so in the log", async () => {
    const { agent, errors } = createAgent();
    const broken = {
      startToolSpan() {},
      endToolSpan() {},
      startSteerSpan() {
        throw new Error("span processor is down");
      },
      endSteerSpan() {},
      recordUsage() {},
      end() {},
    } as unknown as TurnSpanHandle;
    const input = new Pushable<any>();
    const push = vi.spyOn(input, "push");
    agent.sessions["test-session"] = mockSessionState({
      input,
      turnQueue: [
        {
          promptUuid: "running",
          isLocalOnlyCommand: false,
          settled: false,
          resolve: () => {},
          reject: () => {},
          telemetryHandle: broken,
        },
      ],
    });
    await expect(agent.steer(steerRequest(STEER_1))).resolves.toEqual({ outcome: "injected" });
    expect(push).toHaveBeenCalledTimes(1);
    expect(agent.sessions["test-session"].turnQueue![0].steeredEchoes?.size).toBe(1);
    expect(String(errors[0]?.[0])).toContain("could not record a steered message in the trace");
  });

  it("joins a turn that is queued but not yet echoed, and ends with that turn's echoes", async () => {
    const { agent } = createAgent();
    const openingEchoed = Promise.withResolvers<void>();
    const finish = Promise.withResolvers<void>();
    inject(agent, (input) =>
      (async function* () {
        const iter = input[Symbol.asyncIterator]();
        const { value: opening } = await iter.next();
        const { value: steered } = await iter.next();
        await openingEchoed.promise;
        yield userEcho(opening);
        yield userEcho(steered);
        await finish.promise;
        yield successfulResultMessage();
        yield idle;
      })(),
    );
    const turn = startTurn(agent);
    await waitFor(() => (agent.sessions["test-session"]?.turnQueue?.length ?? 0) === 1);
    expect(agent.sessions["test-session"]?.activeTurn ?? null).toBeNull();
    await expect(agent.steer(steerRequest(STEER_1))).resolves.toEqual({ outcome: "injected" });
    openingEchoed.resolve();
    await waitFor(() => finished("steer").length === 1);
    expect(only("steer").attributes["steer.delivered"]).toBe(true);
    finish.resolve();
    await turn;
    expect(only("steer").parentSpanContext?.spanId).toBe(only("agent_turn").spanContext().spanId);
  });

  it("uses a fresh uuid for each steer, which is what the span is keyed by", async () => {
    const { agent } = createAgent();
    const input = new Pushable<any>();
    const push = vi.spyOn(input, "push");
    const handle = service.startTurnSpan("acp-session", OPENING_BLOCKS[1].text);
    agent.sessions["test-session"] = mockSessionState({
      input,
      turnQueue: [
        {
          promptUuid: randomUUID(),
          isLocalOnlyCommand: false,
          settled: false,
          resolve: () => {},
          reject: () => {},
          telemetryHandle: handle,
        },
      ],
    });
    await agent.steer(steerRequest(STEER_1));
    await agent.steer(steerRequest(STEER_2));
    handle.end("OK", "answered");
    const uuids = push.mock.calls.map((call) => call[0].uuid);
    expect(new Set(uuids).size).toBe(2);
    expect(
      finished("steer")
        .map((span) => span.attributes["steer.uuid"])
        .sort(),
    ).toEqual([...uuids].sort());
  });
});
