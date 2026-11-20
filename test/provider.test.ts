import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { normalizeContext } from "@earendil-works/pi-ai/utils/transcript";
import { isContextOverflow } from "@earendil-works/pi-ai";
import type {
	AssistantMessage,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	Message,
	Model,
	Api,
} from "@earendil-works/pi-ai";
import { createAcpProvider, type AcpProviderOptions, type AcpProviderHandle } from "../src/provider.ts";
import { AGENT_PRESETS, type AcpAgentConfig } from "../src/config.ts";
import { toToolResult, type AcpToolDetails } from "../src/native-tools.ts";

const fakePath = fileURLToPath(new URL("./fake-agent.mjs", import.meta.url));

function fakeAgent(overrides: Partial<AcpAgentConfig> = {}): AcpAgentConfig {
	return {
		id: "fake",
		name: "Fake agent",
		command: process.execPath,
		args: [fakePath],
		env: {},
		authMethod: undefined,
		piTools: "off",
		mcpTransport: "stdio",
		systemPrompt: "prompt",
		sessionMeta: undefined,
		ownToolsOffMeta: undefined,
		...overrides,
	};
}

const handles: AcpProviderHandle[] = [];
test.afterEach(() => {
	for (const handle of handles.splice(0)) handle.dispose();
});

function makeProvider(
	agents: AcpAgentConfig[] = [fakeAgent()],
	extra: Omit<AcpProviderOptions, "config"> = {},
): AcpProviderHandle {
	const handle = createAcpProvider({ config: { agents, autoApproveTools: true, handshakeTimeoutMs: 8000, debug: false, nativeTools: false }, ...extra });
	handles.push(handle);
	return handle;
}

function firstModel(handle: AcpProviderHandle): Model<Api> {
	const model = handle.provider.getModels()[0];
	assert.ok(model);
	return model;
}

function userContext(text: string) {
	const messages: Message[] = [{ role: "user", content: [{ type: "text", text }], timestamp: Date.now() }];
	return normalizeContext({ messages });
}

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

async function lastEvent(handle: AcpProviderHandle, text: string, model = firstModel(handle)) {
	return (await collect(handle.provider.streamSimple(model, userContext(text), {}))).at(-1);
}

function messageText(message: AssistantMessage): string {
	return message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("");
}

test("each configured agent is a model of the acp provider", () => {
	const handle = makeProvider([fakeAgent(), fakeAgent({ id: "other", name: "Other agent" })]);
	assert.equal(handle.provider.id, "acp");
	assert.deepEqual(
		handle.provider.getModels().map((model) => [model.provider, model.id, model.name]),
		[
			["acp", "fake", "Fake agent"],
			["acp", "other", "Other agent"],
		],
	);
});

test("happy path: ordered events, balanced blocks, zero usage", async () => {
	const handle = makeProvider();
	const events = await collect(handle.provider.streamSimple(firstModel(handle), userContext("hello there"), {}));
	const types = events.map((event) => event.type);

	assert.equal(types[0], "start");
	assert.equal(types.filter((type) => type === "start" || type === "done" || type === "error").length, 2);
	for (const kind of ["text", "thinking"] as const) {
		assert.equal(
			types.filter((type) => type === `${kind}_start`).length,
			types.filter((type) => type === `${kind}_end`).length,
			`unbalanced ${kind} blocks`,
		);
	}
	const last = events.at(-1);
	assert.equal(last?.type, "done");
	if (last?.type !== "done") return;
	assert.equal(last.message.stopReason, "stop");
	assert.equal(last.message.usage.totalTokens, 0);
	assert.ok(last.message.content.some((block) => block.type === "thinking" && block.thinking.includes("Considering")));
	const text = messageText(last.message);
	assert.match(text, /⚙️ Executing \[grep for secrets\]/);
	assert.match(text, /✓ grep for secrets: 3 matches\n/);
	assert.doesNotMatch(text, /second line/, "tool output is summarized to its first line");
	assert.match(text, /Hello world #\d+/);
});

test("context size comes from usage_update, not the turn's summed usage", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "USAGE please");
	assert.ok(last?.type === "done");
	const { usage } = last.message;
	assert.equal(usage.totalTokens, 1234);
	assert.equal(usage.output, 20);
	assert.equal(usage.input + usage.cacheRead + usage.cacheWrite, 1234);
});

for (const [label, options] of [
	["a live turn", {}],
	["an isolated request", { cacheRetention: "none" }],
] as const) {
	test(`a long tool loop's summed usage is not a context overflow (${label})`, async () => {
		const handle = makeProvider();
		const model = firstModel(handle);
		const events = await collect(handle.provider.streamSimple(model, userContext("LONGLOOP please"), options));
		const last = events.at(-1);
		assert.ok(last?.type === "done");
		const { usage } = last.message;
		assert.equal(usage.totalTokens, 100000);
		assert.equal(usage.output, 9000);
		assert.ok(usage.input + usage.cacheRead + usage.cacheWrite <= 100000);
		assert.ok(usage.input + usage.cacheRead <= model.contextWindow);
		assert.equal(isContextOverflow(last.message, model.contextWindow), false);
	});
}

async function refreshOffline(handle: AcpProviderHandle, stored?: Model<Api>[]): Promise<Model<Api>[]> {
	let published: Model<Api>[] = [];
	await handle.provider.refreshModels?.({
		allowNetwork: false,
		signal: new AbortController().signal,
		...(stored ? { stored: { models: stored, checkedAt: 0 } } : {}),
		publish: async ({ persist, update }) => {
			published = (persist?.models ?? []) as Model<Api>[];
			update?.();
			return true;
		},
	});
	return published;
}

for (const [label, options] of [
	["a live turn", {}],
	["an isolated request", { cacheRetention: "none" }],
] as const) {
	test(`the refreshed catalogue takes the window the agent reported (${label})`, async () => {
		const handle = makeProvider([fakeAgent(), fakeAgent({ id: "other", name: "Other agent" })]);
		const model = firstModel(handle);
		assert.equal(model.contextWindow, 128_000);
		const last = (await collect(handle.provider.streamSimple(model, userContext("USAGE please"), options))).at(-1);
		assert.ok(last?.type === "done");
		const published = await refreshOffline(handle);
		assert.deepEqual(published.map((entry) => [entry.id, entry.contextWindow]), [["fake", 200_000], ["other", 128_000]]);
		assert.deepEqual(handle.provider.getModels().map((entry) => entry.contextWindow), [200_000, 128_000]);
	});
}

test("a window reported for one of the agent's models applies to that model only", async () => {
	const handle = makeProvider([fakeAgent({ env: { FAKE_MODELS: "config" } })]);
	await handle.provider.refreshModels?.({ allowNetwork: true, signal: new AbortController().signal, publish: async ({ update }) => (update?.(), true) });
	const smart = handle.provider.getModels().find((entry) => entry.id === "fake/smart");
	assert.ok(smart);
	assert.ok((await lastEvent(handle, "USAGE please", smart))?.type === "done");
	const published = await refreshOffline(handle);
	assert.deepEqual(
		published.map((entry) => [entry.id, entry.contextWindow]),
		[["fake", 128_000], ["fake/fast", 128_000], ["fake/smart", 200_000]],
	);
});

test("a persisted reported window survives rediscovery", async () => {
	const handle = makeProvider();
	const stored = handle.provider.getModels().map((entry) => ({ ...entry, contextWindow: 200_000 }));
	await handle.provider.refreshModels?.({
		allowNetwork: true,
		signal: new AbortController().signal,
		stored: { models: stored, checkedAt: 0 },
		publish: async ({ update }) => (update?.(), true),
	});
	assert.equal(firstModel(handle).contextWindow, 200_000);
});

for (const size of ["1e999", "0", "-5", "12.5", '"200000"', "null"]) {
	test(`a reported size of ${size} keeps the configured window`, async () => {
		const handle = makeProvider([fakeAgent({ env: { FAKE_USAGE_SIZE: size } })]);
		assert.ok((await lastEvent(handle, "USAGE please"))?.type === "done");
		const published = await refreshOffline(handle);
		assert.deepEqual(published.map((entry) => [entry.id, entry.contextWindow]), [["fake", 128_000]]);
		assert.equal(JSON.parse(JSON.stringify(published))[0].contextWindow, 128_000);
	});
}

test("a persisted window that is not a positive integer is not restored", async () => {
	const handle = makeProvider();
	const stored = handle.provider.getModels().map((entry) => ({ ...entry, contextWindow: null as unknown as number }));
	const published = await refreshOffline(handle, stored);
	assert.deepEqual(published.map((entry) => [entry.id, entry.contextWindow]), [["fake", 128_000]]);
});

async function isolatedSessionParams(agent: AcpAgentConfig) {
	const handle = makeProvider([agent]);
	const last = (await collect(handle.provider.streamSimple(firstModel(handle), userContext("SESSIONINFO"), { cacheRetention: "none" }))).at(-1);
	assert.ok(last?.type === "done");
	return JSON.parse(messageText(last.message)) as { mcpServers: unknown[]; _meta?: unknown };
}

test("an isolated request runs with the agent's own tools off when its preset can turn them off", async () => {
	const claude = AGENT_PRESETS["claude-agent-acp"];
	assert.ok(claude?.ownToolsOffMeta && claude.sessionMeta);
	const params = await isolatedSessionParams(
		fakeAgent({ piTools: "pi", sessionMeta: claude.sessionMeta, ownToolsOffMeta: claude.ownToolsOffMeta }),
	);
	assert.deepEqual(params.mcpServers, []);
	assert.deepEqual(params._meta, {
		claudeCode: { options: { allowedTools: ["mcp__pi"], tools: [], strictMcpConfig: true } },
	});
});

test("an isolated request keeps the session params of an agent without a tools switch", async () => {
	const codex = AGENT_PRESETS["codex-acp"];
	assert.ok(codex && codex.ownToolsOffMeta === undefined);
	const params = await isolatedSessionParams(
		fakeAgent({ piTools: codex.piTools ?? "both", sessionMeta: { vendor: { option: true } }, ownToolsOffMeta: codex.ownToolsOffMeta }),
	);
	assert.deepEqual(params.mcpServers, []);
	assert.equal(params._meta, undefined);
});

test("without usage_update the turn's summed usage is reported unchanged", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "LONGLOOP NOUPDATE please");
	assert.ok(last?.type === "done");
	const { usage } = last.message;
	assert.deepEqual([usage.input, usage.output, usage.cacheRead, usage.cacheWrite], [2000, 9000, 4869752, 120000]);
	assert.equal(usage.totalTokens, 5000752);
});

test("agents run as separate processes with separate sessions", async () => {
	const handle = makeProvider([fakeAgent(), fakeAgent({ id: "other", name: "Other agent" })]);
	const [fake, other] = handle.provider.getModels();
	assert.ok(fake && other);
	const first = await lastEvent(handle, "one", fake);
	const second = await lastEvent(handle, "two", other);
	assert.ok(first?.type === "done" && second?.type === "done");
	// Each fake process counts its own turns, so both start at #0.
	assert.match(messageText(first.message), /#0/);
	assert.match(messageText(second.message), /#0/);
	assert.equal(second.message.model, "other");
});

/** Streams a SLOW prompt with `first`, and once it is under way an ECHO prompt with `second`; both to completion. */
async function overlapping(
	handle: AcpProviderHandle,
	first: { model: Model<Api>; sessionId?: string },
	second: { model: Model<Api>; sessionId?: string },
) {
	let started: () => void = () => {};
	const underWay = new Promise<void>((resolve) => (started = resolve));
	const slow = (async () => {
		const events: AssistantMessageEvent[] = [];
		for await (const event of handle.provider.streamSimple(first.model, userContext("SLOW one"), first.sessionId ? { sessionId: first.sessionId } : {})) {
			events.push(event);
			if (event.type === "thinking_delta") started();
		}
		started();
		return events.at(-1);
	})();
	await underWay;
	const echo = (await collect(handle.provider.streamSimple(second.model, userContext("ECHO two"), second.sessionId ? { sessionId: second.sessionId } : {}))).at(-1);
	return { slow: await slow, echo };
}

for (const sessionId of [undefined, "conv"]) {
	test(`concurrent streams on two agents both complete with their own output (${sessionId ? "same session id" : "no session id"})`, async () => {
		const handle = makeProvider([fakeAgent(), fakeAgent({ id: "other", name: "Other agent" })]);
		const [fake, other] = handle.provider.getModels();
		assert.ok(fake && other);
		const { slow, echo } = await overlapping(handle, { model: fake, ...(sessionId ? { sessionId } : {}) }, { model: other, ...(sessionId ? { sessionId } : {}) });
		assert.ok(slow?.type === "done", `the first stream ends with ${slow?.type}`);
		assert.match(messageText(slow.message), /Hello world #0/);
		assert.doesNotMatch(messageText(slow.message), /ECHO/);
		assert.ok(echo?.type === "done");
		assert.match(messageText(echo.message), /ECHO\[sess-1\]<<<.*ECHO two/s);
	});
}

test("a request replacing one paused in onPayload owns the turn, and the paused one ends aborted while its hook still waits", async () => {
	const handle = makeProvider();
	const model = firstModel(handle);
	let release: () => void = () => {};
	const gate = new Promise<void>((resolve) => (release = resolve));
	let paused: () => void = () => {};
	const reached = new Promise<void>((resolve) => (paused = resolve));
	const first = collect(
		handle.provider.streamSimple(model, userContext("ECHO first"), {
			sessionId: "conv",
			onPayload: async () => {
				paused();
				await gate;
				return undefined;
			},
		}),
	);
	await reached;
	let started: () => void = () => {};
	const underWay = new Promise<void>((resolve) => (started = resolve));
	const second = (async () => {
		const events: AssistantMessageEvent[] = [];
		for await (const event of handle.provider.streamSimple(model, userContext("SLOW second"), { sessionId: "conv" })) {
			events.push(event);
			if (event.type === "thinking_delta") started();
		}
		started();
		return events.at(-1);
	})();
	await underWay;
	const ended = await Promise.race([first, new Promise<undefined>((resolve) => setTimeout(() => resolve(undefined), 2000))]);
	release();
	const firstLast = ended?.at(-1);
	assert.ok(firstLast?.type === "error", `the replaced request ends with ${firstLast?.type ?? "nothing while its hook waits"}`);
	assert.equal(firstLast.reason, "aborted", firstLast.error.errorMessage ?? "");
	const secondLast = await second;
	assert.ok(secondLast?.type === "done", `the replacing request ends with ${secondLast?.type}`);
	assert.match(messageText(secondLast.message), /Hello world #0/);
});

test("concurrent streams on two conversations of one agent run in separate sessions", async () => {
	const handle = makeProvider();
	const model = firstModel(handle);
	const { slow, echo } = await overlapping(handle, { model, sessionId: "conv-a" }, { model, sessionId: "conv-b" });
	assert.ok(slow?.type === "done", `the first stream ends with ${slow?.type}`);
	assert.match(messageText(slow.message), /Hello world #0/);
	assert.doesNotMatch(messageText(slow.message), /ECHO/);
	assert.ok(echo?.type === "done");
	const echoSession = /ECHO\[(sess-\d+)\]<<<.*ECHO two/s.exec(messageText(echo.message))?.[1];
	assert.ok(echoSession);
	const again = (await collect(handle.provider.streamSimple(model, userContext("ECHO three"), { sessionId: "conv-a" }))).at(-1);
	assert.ok(again?.type === "done");
	const slowSession = /ECHO\[(sess-\d+)\]/.exec(messageText(again.message))?.[1];
	assert.ok(slowSession);
	assert.notEqual(slowSession, echoSession, "each conversation keeps its own ACP session");
});

test("each agent's session gets Pi's instructions even when session ids collide", async () => {
	const handle = makeProvider([fakeAgent(), fakeAgent({ id: "other", name: "Other agent" })]);
	const [fake, other] = handle.provider.getModels();
	assert.ok(fake && other);
	const messages: Message[] = [{ role: "user", content: [{ type: "text", text: "ECHO hi" }], timestamp: Date.now() }];
	const context = normalizeContext({ systemPrompt: "Be brief.", messages });
	for (const model of [fake, other]) {
		const last = (await collect(handle.provider.streamSimple(model, context, {}))).at(-1);
		assert.ok(last?.type === "done");
		assert.match(messageText(last.message), /ECHO\[sess-1\]<<<.*<system-instructions>/s, `${model.id} gets the system prompt`);
	}
});

test("a tool call reported already completed renders its result", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "ONESHOT go");
	assert.ok(last?.type === "done");
	assert.match(messageText(last.message), /⚙️ Executing \[read notes\]\.\.\.\n✓ read notes: 2 notes\n/);
});

test("agentFor resolves bare and agent/model ids to their agent", () => {
	const handle = makeProvider([fakeAgent(), fakeAgent({ id: "other", name: "Other agent" })]);
	assert.equal(handle.agentFor("fake"), "fake");
	assert.equal(handle.agentFor("other/smart"), "other");
	assert.equal(handle.agentFor("ghost/smart"), undefined);
	assert.equal(handle.agentFor("fakeish"), undefined);
});

test("a model id without a configured agent fails cleanly", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "hi", { ...firstModel(handle), id: "ghost" });
	assert.equal(last?.type, "error");
	if (last?.type !== "error") return;
	assert.match(last.error.errorMessage ?? "", /Unknown ACP agent 'ghost'/);
});

test("abort mid-stream finalizes as aborted and the session survives", async () => {
	const handle = makeProvider();
	const model = firstModel(handle);
	const controller = new AbortController();
	const events: AssistantMessageEvent[] = [];
	for await (const event of handle.provider.streamSimple(model, userContext("SLOW abort me"), { signal: controller.signal })) {
		events.push(event);
		if (event.type === "thinking_delta") controller.abort();
	}
	const last = events.at(-1);
	assert.equal(last?.type, "error");
	if (last?.type !== "error") return;
	assert.equal(last.reason, "aborted");
	assert.equal(last.error.stopReason, "aborted");
	assert.equal((await lastEvent(handle, "still here"))?.type, "done");
});

test("after a reset, a restarted agent's session that reuses an old id gets the instructions and history again", async () => {
	const handle = makeProvider();
	const messages: Message[] = [{ role: "user", content: [{ type: "text", text: "ECHO one" }], timestamp: Date.now() }];
	const run = async () =>
		(await collect(handle.provider.streamSimple(firstModel(handle), normalizeContext({ systemPrompt: "SYSTEM-RULES", messages }), {}))).at(-1);
	const first = await run();
	assert.ok(first?.type === "done");
	assert.match(messageText(first.message), /^ECHO\[sess-1\]<<<<system-instructions>/);
	handle.resetSession(process.cwd());
	// WHY: the fake numbers sessions per process, so once it restarts its next session is sess-1 again.
	assert.equal((await lastEvent(handle, "CRASH out"))?.type, "error");
	messages.push(first.message, { role: "user", content: [{ type: "text", text: "ECHO two" }], timestamp: Date.now() + 1 });
	const second = await run();
	assert.ok(second?.type === "done", JSON.stringify(second));
	assert.match(messageText(second.message), /^ECHO\[sess-1\]<<<<system-instructions>[\s\S]*SYSTEM-RULES[\s\S]*<conversation-history>[\s\S]*ECHO one[\s\S]*ECHO two>>>$/);
});

test("process crash surfaces as an error and the next prompt recovers", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "CRASH out");
	assert.equal(last?.type, "error");
	if (last?.type !== "error") return;
	assert.match(last.error.errorMessage ?? "", /Fake agent exited unexpectedly/);
	assert.equal((await lastEvent(handle, "after crash"))?.type, "done");
});

test("auth_required lists the agent's advertised auth methods", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "AUTH please");
	assert.equal(last?.type, "error");
	if (last?.type !== "error") return;
	assert.match(last.error.errorMessage ?? "", /Fake agent requires authentication/);
	assert.match(last.error.errorMessage ?? "", /'fake-login' \(Fake login: Pretend to sign in\)/);
});

test("a configured auth method signs in before the session is created", async () => {
	const handle = makeProvider([fakeAgent({ env: { FAKE_AUTH: "1" }, authMethod: "fake-login" })]);
	assert.equal((await lastEvent(handle, "hello"))?.type, "done");
});

test("RPC errors surface as stream errors", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "RPCERROR go");
	assert.equal(last?.type, "error");
	if (last?.type !== "error") return;
	assert.match(last.error.errorMessage ?? "", /fake rpc failure/);
});

test("a context without a user message fails before start", async () => {
	const handle = makeProvider();
	const events = await collect(handle.provider.streamSimple(firstModel(handle), normalizeContext({ messages: [] }), {}));
	assert.equal(events.length, 1);
	assert.equal(events[0]?.type, "error");
});

test("max_tokens maps to the length stop reason", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "MAXTOKENS hit");
	assert.equal(last?.type, "done");
	if (last?.type !== "done") return;
	assert.equal(last.reason, "length");
});

test("refusal stop reason surfaces as an error", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "REFUSAL please");
	assert.equal(last?.type, "error");
	if (last?.type !== "error") return;
	assert.match(last.error.errorMessage ?? "", /Fake agent refused/);
});

test("instrumentation hooks are honored", async () => {
	const handle = makeProvider();
	const payloads: unknown[] = [];
	const raws: unknown[] = [];
	const events = await collect(
		handle.provider.streamSimple(firstModel(handle), userContext("hook me"), {
			onPayload: (payload) => {
				payloads.push(payload);
				return undefined;
			},
			onProviderStreamEvent: (data) => {
				raws.push(data);
			},
		}),
	);
	assert.equal(events.at(-1)?.type, "done");
	assert.equal(payloads.length, 1);
	assert.match(JSON.stringify(payloads[0]), /"text":"hook me"/);
	assert.ok(raws.length >= 3);
});

test("a rejecting permission policy reaches the agent", async () => {
	const seen: string[] = [];
	const handle = makeProvider([fakeAgent()], {
		permissionPolicy: async (request, { agent }) => {
			seen.push(`${agent.id}:${request.title}`);
			return { outcome: "selected", optionId: "deny" };
		},
	});
	const last = await lastEvent(handle, "PERMISSION ask");
	assert.equal(last?.type, "done");
	if (last?.type !== "done") return;
	assert.match(messageText(last.message), /\[approved=deny\]/);
	assert.deepEqual(seen, ["fake:rm -rf build"]);
});

for (const [label, env, expected] of [
	["rejected", {}, /\[approved=deny\]/],
	["cancelled when it offers no reject option", { FAKE_PERMISSION_NO_REJECT: "1" }, /\[approved=cancelled\]/],
] as const) {
	test(`a permission request from an isolated request of an agent without a tools switch is ${label}, without the policy`, async () => {
		const codex = AGENT_PRESETS["codex-acp"];
		assert.ok(codex && codex.ownToolsOffMeta === undefined);
		const seen: string[] = [];
		const handle = makeProvider([fakeAgent({ piTools: codex.piTools ?? "both", ownToolsOffMeta: undefined, env })], {
			permissionPolicy: async (request) => {
				seen.push(request.title);
				return { outcome: "selected", optionId: "allow" };
			},
		});
		const last = (
			await collect(handle.provider.streamSimple(firstModel(handle), userContext("PERMISSION summarize"), { cacheRetention: "none" }))
		).at(-1);
		assert.ok(last?.type === "done");
		assert.match(messageText(last.message), expected);
		assert.deepEqual(seen, []);
		const live = await lastEvent(handle, "PERMISSION ask");
		assert.ok(live?.type === "done");
		assert.match(messageText(live.message), /\[approved=allow\]/, "live sessions still ask the policy");
		assert.deepEqual(seen, ["rm -rf build"]);
	});
}

test("aborting while a permission prompt is open answers it as cancelled", async () => {
	const controller = new AbortController();
	let policySignal: AbortSignal | undefined;
	const handle = makeProvider([fakeAgent()], {
		permissionPolicy: (_request, { signal }) => {
			policySignal = signal;
			controller.abort();
			// An interactive dialog that stays open until dismissed by the signal.
			return new Promise((resolve) =>
				signal?.addEventListener("abort", () => resolve({ outcome: "cancelled" }), { once: true }),
			);
		},
	});
	const events = await collect(
		handle.provider.streamSimple(firstModel(handle), userContext("PERMISSION then abort"), { signal: controller.signal }),
	);
	assert.equal(events.at(-1)?.type, "error");
	assert.ok(policySignal?.aborted);
	assert.equal((await lastEvent(handle, "next turn"))?.type, "done");
});

function modelsAgent(mode: "config" | "legacy"): AcpAgentConfig {
	return fakeAgent({ env: { FAKE_MODELS: mode } });
}

function storeContext(stored?: { models: readonly unknown[] }) {
	let published: { models: readonly unknown[] } | undefined;
	const context = {
		allowNetwork: true,
		signal: new AbortController().signal,
		...(stored ? { stored } : {}),
		publish: async (publication: { persist?: { models: readonly unknown[] } | null; update?: () => void }) => {
			published = publication.persist ?? undefined;
			publication.update?.();
			return true;
		},
	};
	return { context: context as never, persisted: () => published };
}

test("discovery lists the agent's models, grouped options flattened", async () => {
	const handle = makeProvider([modelsAgent("config")]);
	const { context, persisted } = storeContext();
	await handle.provider.refreshModels?.(context);
	const models = handle.provider.getModels();
	assert.deepEqual(
		models.map((model) => [model.id, model.name]),
		[
			["fake", "Fake agent"],
			["fake/fast", "Fast (Fake agent)"],
			["fake/smart", "Smart (Fake agent)"],
		],
	);
	assert.equal(models[1]?.reasoning, true);
	assert.deepEqual(models[1]?.thinkingLevelMap, {
		off: "none",
		minimal: null,
		low: "low",
		medium: "medium",
		high: "high",
		xhigh: null,
		max: null,
	});
	assert.equal(persisted()?.models.length, 3);
});

test("picking a discovered model and thinking level switches the session", async () => {
	const handle = makeProvider([modelsAgent("config")]);
	await handle.provider.refreshModels?.(storeContext().context);
	const smart = handle.provider.getModels().find((model) => model.id === "fake/smart");
	assert.ok(smart);
	const messages: Message[] = [{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() }];
	const first = await collect(handle.provider.streamSimple(smart, normalizeContext({ messages }), { reasoning: "high" }));
	const done = first.at(-1);
	assert.ok(done?.type === "done");
	assert.match(messageText(done.message), /\[model=smart thought=high\]/);
	// The bare agent model goes back to the agent's default in the same conversation; no reasoning means off.
	messages.push(done.message, { role: "user", content: [{ type: "text", text: "again" }], timestamp: Date.now() + 1 });
	const back = (await collect(handle.provider.streamSimple(firstModel(handle), normalizeContext({ messages }), {}))).at(-1);
	assert.ok(back?.type === "done");
	assert.match(messageText(back.message), /\[model=fast thought=none\]/);
});

test("a reasoning option in the \"thinking\" category (Mistral Vibe) gives thinking levels", async () => {
	const handle = makeProvider([fakeAgent({ env: { FAKE_MODELS: "config", FAKE_THOUGHT_CATEGORY: "thinking" } })]);
	await handle.provider.refreshModels?.(storeContext().context);
	const smart = handle.provider.getModels().find((model) => model.id === "fake/smart");
	assert.ok(smart?.reasoning);
	assert.equal(smart.thinkingLevelMap?.high, "high");
	const messages: Message[] = [{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() }];
	const done = (await collect(handle.provider.streamSimple(smart, normalizeContext({ messages }), { reasoning: "high" }))).at(-1);
	assert.ok(done?.type === "done");
	assert.match(messageText(done.message), /\[model=smart thought=high\]/);
});

test("each agent's bare model keeps its own session default, even when session ids collide", async () => {
	const handle = makeProvider([
		fakeAgent({ env: { FAKE_MODELS: "config", FAKE_DEFAULT_MODEL: "smart" } }),
		fakeAgent({ id: "other", name: "Other agent", env: { FAKE_MODELS: "config" } }),
	]);
	const [fake, other] = handle.provider.getModels();
	assert.ok(fake && other);
	// Both fake processes name their first session `sess-1`.
	const first = await lastEvent(handle, "hi", fake);
	const second = await lastEvent(handle, "hi", other);
	assert.ok(first?.type === "done" && second?.type === "done");
	assert.match(messageText(first.message), /\[model=smart /);
	assert.match(messageText(second.message), /\[model=fast /);
});

test("legacy models API is discovered and switched with session/set_model", async () => {
	const handle = makeProvider([modelsAgent("legacy")]);
	await handle.provider.refreshModels?.(storeContext().context);
	const ids = handle.provider.getModels().map((model) => model.id);
	assert.deepEqual(ids, ["fake", "fake/fast", "fake/smart"]);
	const smart = handle.provider.getModels()[2];
	assert.ok(smart && !smart.reasoning);
	const last = await lastEvent(handle, "hi", smart);
	assert.ok(last?.type === "done");
	assert.match(messageText(last.message), /\[model=smart/);
});

test("a model the agent does not offer fails with the available list", async () => {
	const handle = makeProvider([modelsAgent("config")]);
	const last = await lastEvent(handle, "hi", { ...firstModel(handle), id: "fake/ghost" });
	assert.ok(last?.type === "error");
	assert.match(last.error.errorMessage ?? "", /does not offer model 'ghost'. Available: fast, smart/);
});

test("stored models are restored without spawning when network is not allowed", async () => {
	const discovered = makeProvider([modelsAgent("config")]);
	const first = storeContext();
	await discovered.provider.refreshModels?.(first.context);
	const stored = first.persisted();
	assert.ok(stored);

	const offline = makeProvider([fakeAgent({ command: "/nonexistent/agent" })]);
	const { context } = storeContext(stored);
	await offline.provider.refreshModels?.({ ...(context as object), allowNetwork: false } as never);
	assert.deepEqual(
		offline.provider.getModels().map((model) => model.id),
		["fake", "fake/fast", "fake/smart"],
	);
	assert.equal(offline.provider.getModels()[1]?.thinkingLevelMap?.high, "high");
});

test("agents without model support list only their default model after discovery", async () => {
	const handle = makeProvider();
	await handle.provider.refreshModels?.(storeContext().context);
	assert.deepEqual(handle.provider.getModels().map((model) => model.id), ["fake"]);
});

test("an agent slash command sent as the prompt returns the agent's output", async () => {
	const handle = makeProvider();
	const last = await lastEvent(handle, "/context");
	assert.ok(last?.type === "done");
	assert.equal(messageText(last.message), "context: 42%");
	assert.deepEqual((await handle.commands("fake")).map((command) => command.name), ["context", "session"]);
});

/** Plays Pi's agent loop: answers each mirrored tool call and calls the provider again until the turn ends. */
async function nativeTurn(handle: AcpProviderHandle, text: string) {
	const model = firstModel(handle);
	const messages: Message[] = [{ role: "user", content: [{ type: "text", text }], timestamp: Date.now() }];
	const finals: AssistantMessageEvent[] = [];
	const results: { args: unknown; details: AcpToolDetails; isError: boolean }[] = [];
	for (let step = 0; step < 10; step++) {
		const last = (await collect(handle.provider.streamSimple(model, normalizeContext({ messages }), {}))).at(-1);
		assert.ok(last);
		finals.push(last);
		if (last.type !== "done" || last.reason !== "toolUse") return { finals, results };
		messages.push(last.message);
		for (const block of last.message.content) {
			if (block.type !== "toolCall") continue;
			const result = toToolResult(await handle.awaitTool(block.id, undefined, () => {}));
			results.push({ args: block.arguments, details: result.details, isError: result.isError === true });
			messages.push({
				role: "toolResult",
				toolCallId: block.id,
				toolName: block.name,
				content: result.content,
				details: result.details,
				isError: result.isError === true,
				timestamp: Date.now(),
			} as Message);
		}
	}
	assert.fail("the turn did not end");
}

test("native tools: a tool call ends the message and the turn continues after its result", async () => {
	const handle = makeProvider(undefined, { nativeTools: () => true });
	const { finals, results } = await nativeTurn(handle, "hello");
	assert.deepEqual(
		finals.map((event) => (event.type === "done" ? event.reason : event.type)),
		["toolUse", "stop"],
	);
	const first = finals[0];
	assert.ok(first?.type === "done");
	assert.ok(first.message.content.some((block) => block.type === "thinking"));
	const call = first.message.content.at(-1);
	assert.ok(call?.type === "toolCall");
	assert.equal(call.name, "acp_tool");
	assert.deepEqual(call.arguments, { agent: "fake", title: "grep for secrets", kind: "search" });
	assert.equal(results[0]?.details.status, "completed");
	assert.equal(results[0]?.details.output, "3 matches\nsecond line");
	const last = finals[1];
	assert.ok(last?.type === "done");
	assert.doesNotMatch(messageText(last.message), /Executing/);
	assert.match(messageText(last.message), /^Hello world #\d+$/);
});

test("native tools: each new tool call gets its own message", async () => {
	const handle = makeProvider(undefined, { nativeTools: () => true });
	const { finals, results } = await nativeTurn(handle, "ONESHOT go");
	assert.deepEqual(
		finals.map((event) => (event.type === "done" ? event.reason : event.type)),
		["toolUse", "toolUse", "stop"],
	);
	assert.deepEqual(
		results.map((result) => (result.args as { title: string }).title),
		["read notes", "grep for secrets"],
	);
});

test("native tools: an edit returns a display diff", async () => {
	const handle = makeProvider(undefined, { nativeTools: () => true });
	const { results } = await nativeTurn(handle, "DIFF go");
	const edit = results[0];
	assert.ok(edit);
	assert.deepEqual((edit.args as { locations: unknown }).locations, [{ path: "/tmp/notes.txt", line: 2 }]);
	assert.equal(edit.details.diffs[0]?.path, "/tmp/notes.txt");
	assert.match(edit.details.diffs[0]?.diff ?? "", /-.*two/);
	assert.match(edit.details.diffs[0]?.diff ?? "", /\+.*TWO/);
});

test("native tools: aborting while Pi waits on a mirrored call aborts the turn", async () => {
	const handle = makeProvider(undefined, { nativeTools: () => true });
	const model = firstModel(handle);
	const first = (await collect(handle.provider.streamSimple(model, userContext("SLOW go"), {}))).at(-1);
	assert.ok(first?.type === "done" && first.reason === "toolUse");
	const call = first.message.content.at(-1);
	assert.ok(call?.type === "toolCall");
	const controller = new AbortController();
	const waiting = handle.awaitTool(call.id, controller.signal, () => {});
	controller.abort();
	await assert.rejects(waiting, /aborted/);
});

test("native tools: two conversations' mirrored calls get distinct ids and stay with their own turn", async () => {
	const handle = makeProvider(undefined, { nativeTools: () => true });
	const model = firstModel(handle);
	const opening = async (text: string, sessionId: string) => {
		const messages: Message[] = [{ role: "user", content: [{ type: "text", text }], timestamp: Date.now() }];
		const last = (await collect(handle.provider.streamSimple(model, normalizeContext({ messages }), { sessionId }))).at(-1);
		assert.ok(last?.type === "done" && last.reason === "toolUse");
		const call = last.message.content.at(-1);
		assert.ok(call?.type === "toolCall");
		return { messages: [...messages, last.message], call };
	};
	const a = await opening("go a", "conv-a");
	const b = await opening("go b", "conv-b");
	assert.notEqual(a.call.id, b.call.id, "the agent reported both calls as t1");
	const controller = new AbortController();
	const waiting = handle.awaitTool(b.call.id, controller.signal, () => {});
	controller.abort();
	await assert.rejects(waiting, /aborted/);
	const result = toToolResult(await handle.awaitTool(a.call.id, undefined, () => {}));
	assert.equal(result.details.output, "3 matches\nsecond line");
	a.messages.push({
		role: "toolResult",
		toolCallId: a.call.id,
		toolName: a.call.name,
		content: result.content,
		details: result.details,
		isError: false,
		timestamp: Date.now(),
	} as Message);
	const last = (await collect(handle.provider.streamSimple(model, normalizeContext({ messages: a.messages }), { sessionId: "conv-a" }))).at(-1);
	assert.ok(last?.type === "done", `conversation a ends with ${last?.type}`);
	assert.match(messageText(last.message), /^Hello world #\d+$/);
});

test("a prompt prepared before the agent restarted is not sent into another conversation's new session of the same id", async () => {
	const handle = makeProvider();
	const model = firstModel(handle);
	let release: () => void = () => {};
	const gate = new Promise<void>((resolve) => (release = resolve));
	let paused: () => void = () => {};
	const reached = new Promise<void>((resolve) => (paused = resolve));
	const held = collect(
		handle.provider.streamSimple(model, userContext("ECHO held"), {
			sessionId: "conv-a",
			onPayload: async () => {
				paused();
				await gate;
				return undefined;
			},
		}),
	);
	await reached;
	const crashed = (await collect(handle.provider.streamSimple(model, userContext("CRASH now"), { sessionId: "conv-b" }))).at(-1);
	assert.equal(crashed?.type, "error");
	const fresh = (await collect(handle.provider.streamSimple(model, userContext("ECHO fresh"), { sessionId: "conv-c" }))).at(-1);
	assert.ok(fresh?.type === "done");
	assert.match(messageText(fresh.message), /^ECHO\[sess-1\]<<</, "the respawned agent reuses the held request's session id");
	release();
	const last = (await held).at(-1);
	assert.ok(last?.type === "error", `the held request ends with ${last?.type}: ${last?.type === "done" ? messageText(last.message) : ""}`);
	assert.match(last.error.errorMessage ?? "", /session sess-1 ended while the request was being prepared/);
	const after = (await collect(handle.provider.streamSimple(model, userContext("ECHO after"), { sessionId: "conv-c" }))).at(-1);
	assert.ok(after?.type === "done");
	assert.doesNotMatch(messageText(after.message), /ECHO held/);
});

test("a replacement's model is applied after a superseded request's settings change lands", async () => {
	const handle = makeProvider([fakeAgent({ env: { FAKE_MODELS: "config", FAKE_SETTING_DELAY: "300" } })]);
	await handle.provider.refreshModels?.(storeContext().context);
	const smart = handle.provider.getModels().find((candidate) => candidate.id === "fake/smart");
	assert.ok(smart);
	const bare = firstModel(handle);
	const messages: Message[] = [{ role: "user", content: [{ type: "text", text: "hi" }], timestamp: Date.now() }];
	const warm = (await collect(handle.provider.streamSimple(bare, normalizeContext({ messages }), { sessionId: "conv" }))).at(-1);
	assert.ok(warm?.type === "done");
	assert.match(messageText(warm.message), /\[model=fast /);
	messages.push(warm.message, { role: "user", content: [{ type: "text", text: "SLOW again" }], timestamp: Date.now() + 1 });
	const superseded = collect(handle.provider.streamSimple(smart, normalizeContext({ messages }), { sessionId: "conv" }));
	await new Promise((resolve) => setTimeout(resolve, 100));
	// WHY: the fake reads its model when a prompt starts, so the replacement's prompt waits until the
	// superseded change (sent at about 0 ms, answered at 300 ms) has landed.
	const replacement = (
		await collect(
			handle.provider.streamSimple(bare, normalizeContext({ messages }), {
				sessionId: "conv",
				onPayload: () => new Promise<undefined>((resolve) => setTimeout(() => resolve(undefined), 400)),
			}),
		)
	).at(-1);
	assert.equal((await superseded).at(-1)?.type, "error");
	assert.ok(replacement?.type === "done", `the replacement ends with ${replacement?.type}`);
	assert.match(messageText(replacement.message), /\[model=fast /);
});

test("a hook that aborts its request and rejects ends the stream as aborted without an unhandled rejection", async () => {
	const handle = makeProvider();
	const unhandled: unknown[] = [];
	const record = (reason: unknown): void => {
		unhandled.push(reason);
	};
	process.on("unhandledRejection", record);
	try {
		const controller = new AbortController();
		const last = (
			await collect(
				handle.provider.streamSimple(firstModel(handle), userContext("ECHO hook"), {
					signal: controller.signal,
					onPayload: () => {
						controller.abort();
						return Promise.reject(new Error("hook failed"));
					},
				}),
			)
		).at(-1);
		assert.ok(last?.type === "error");
		assert.equal(last.reason, "aborted");
		await new Promise((resolve) => setTimeout(resolve, 50));
		assert.deepEqual(unhandled.map(String), []);
	} finally {
		process.off("unhandledRejection", record);
	}
});
