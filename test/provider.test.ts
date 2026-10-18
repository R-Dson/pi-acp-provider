import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { normalizeContext } from "@earendil-works/pi-ai/utils/transcript";
import type {
	AssistantMessage,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	Message,
	Model,
	Api,
} from "@earendil-works/pi-ai";
import { createAcpProvider, type AcpProviderOptions, type AcpProviderHandle } from "../src/provider.ts";
import type { AcpAgentConfig } from "../src/config.ts";

const fakePath = fileURLToPath(new URL("./fake-agent.mjs", import.meta.url));

function fakeAgent(overrides: Partial<AcpAgentConfig> = {}): AcpAgentConfig {
	return {
		id: "fake",
		name: "Fake agent",
		command: process.execPath,
		args: [fakePath],
		env: {},
		authMethod: undefined,
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
	const handle = createAcpProvider({ config: { agents, autoApproveTools: true, handshakeTimeoutMs: 8000 }, ...extra });
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
