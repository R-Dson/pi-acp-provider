import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { AcpClient, AcpRpcError, type AcpClientOptions, type ServerRequestHandler } from "../src/acp-client.ts";
import { autoApprove, parsePermissionRequest, parseRpcMessage, type AcpSessionUpdate } from "../src/acp.ts";

const fakePath = fileURLToPath(new URL("./fake-agent.mjs", import.meta.url));

const clients: AcpClient[] = [];
test.afterEach(() => {
	for (const client of clients.splice(0)) client.dispose();
});

const approveAll: ServerRequestHandler = async ({ params }) => {
	const request = parsePermissionRequest(params);
	assert.ok(request);
	return { outcome: autoApprove(request) };
};

function makeClient(
	overrides: Partial<Omit<AcpClientOptions, "agent">> & { agent?: Partial<AcpClientOptions["agent"]> } = {},
): AcpClient {
	const { agent, ...rest } = overrides;
	const client = new AcpClient({
		agent: { name: "Fake agent", command: process.execPath, args: [fakePath], env: {}, authMethod: undefined, ...agent },
		cwd: process.cwd(),
		handshakeTimeoutMs: 8000,
		onServerRequest: approveAll,
		...rest,
	});
	clients.push(client);
	return client;
}

const promptPayload = (sessionId: string, text: string) => ({ sessionId, prompt: [{ type: "text", text }] });

async function runPrompt(client: AcpClient, text: string, signal?: AbortSignal) {
	const sessionId = await client.ensureSession();
	const updates: AcpSessionUpdate[] = [];
	const result = await client.prompt({
		sessionId,
		payload: promptPayload(sessionId, text),
		onUpdate: (update) => updates.push(update),
		signal,
	});
	const reply = updates.flatMap((update) => (update.type === "agent_message_chunk" ? [update.text] : [])).join("");
	return { result, updates, reply };
}

test("initialize handshake reaches ready", async () => {
	const client = makeClient();
	const info = await client.ensureReady();
	assert.equal(client.state, "ready");
	assert.match(info.name, /^fake-agent-/);
	assert.deepEqual(info.authMethods, [{ id: "fake-login", name: "Fake login", description: "Pretend to sign in" }]);
});

test("session is created once and reused", async () => {
	const client = makeClient();
	const [first, second] = await Promise.all([client.ensureSession(), client.ensureSession()]);
	assert.equal(first, second);
	client.resetSession(process.cwd());
	assert.notEqual(await client.ensureSession(), first);
});

test("spec-shaped session updates stream in order and the turn finishes", async () => {
	const client = makeClient();
	const { result, updates, reply } = await runPrompt(client, "do the thing");
	assert.deepEqual(result, { kind: "finished", stopReason: "end_turn" });
	assert.deepEqual(
		updates.map((update) => update.type),
		["agent_thought_chunk", "tool_call", "tool_call_update", "tool_call_update", "agent_message_chunk", "agent_message_chunk"],
	);
	const completed = updates.find((update) => update.type === "tool_call_update" && update.status === "completed");
	assert.ok(completed?.type === "tool_call_update");
	assert.equal(completed.summary, "3 matches\nsecond line");
	assert.equal(reply, "Hello world #0");
});

test("split NDJSON frames are reassembled", async () => {
	const client = makeClient({ agent: { env: { FAKE_SPLIT: "1" } } });
	const { result, reply } = await runPrompt(client, "split frames");
	assert.equal(result.kind, "finished");
	assert.equal(reply, "Hello world #0");
});

test("permission requests with string ids are answered through the handler", async () => {
	const methods: string[] = [];
	const client = makeClient({
		onServerRequest: async (request) => {
			methods.push(request.method);
			return approveAll(request);
		},
	});
	const { reply } = await runPrompt(client, "PERMISSION needed");
	assert.deepEqual(methods, ["session/request_permission"]);
	assert.match(reply, /\[approved=allow\]/);
});

test("abort sends session/cancel and keeps the session usable", async () => {
	const client = makeClient();
	const sessionId = await client.ensureSession();
	const controller = new AbortController();
	const result = await client.prompt({
		sessionId,
		payload: promptPayload(sessionId, "SLOW work"),
		signal: controller.signal,
		onUpdate: () => controller.abort(),
	});
	assert.deepEqual(result, { kind: "aborted" });
	const second = await runPrompt(client, "quick follow-up");
	assert.equal(second.result.kind, "finished");
	// Late updates of the cancelled turn must not leak into the next one.
	assert.equal(second.reply, "Hello world #0");
});

test("process crash rejects the prompt and the client respawns on next use", async () => {
	const client = makeClient();
	const firstInfo = await client.ensureReady();
	await assert.rejects(runPrompt(client, "CRASH now"), /exited unexpectedly/);
	assert.equal(client.state, "uninitialized");
	const secondInfo = await client.ensureReady();
	assert.notEqual(secondInfo.name, firstInfo.name);
	const { result } = await runPrompt(client, "alive again");
	assert.equal(result.kind, "finished");
});

test("auth_required without a configured method surfaces the RPC error", async () => {
	const client = makeClient({ agent: { env: { FAKE_AUTH: "1" } } });
	await assert.rejects(client.ensureSession(), (error: unknown) => error instanceof AcpRpcError && error.code === -32000);
});

test("auth_required triggers authenticate with the configured method, then retries", async () => {
	const client = makeClient({ agent: { env: { FAKE_AUTH: "1" }, authMethod: "fake-login" } });
	const { result } = await runPrompt(client, "after login");
	assert.equal(result.kind, "finished");
});

test("missing command yields a not-found hint naming the agent", async () => {
	const client = makeClient({ agent: { command: "acp-missing-binary-xyz", args: [] } });
	await assert.rejects(client.ensureReady(), /Fake agent: command 'acp-missing-binary-xyz' not found/);
	assert.equal(client.state, "uninitialized");
});

test("concurrent prompt is rejected while one is in flight", async () => {
	const client = makeClient();
	const sessionId = await client.ensureSession();
	const first = client.prompt({ sessionId, payload: promptPayload(sessionId, "SLOW first"), onUpdate: () => {} });
	await assert.rejects(
		client.prompt({ sessionId, payload: promptPayload(sessionId, "second"), onUpdate: () => {} }),
		/already in flight/,
	);
	await first;
});

test("disposed client refuses new work", async () => {
	const client = makeClient();
	await client.ensureReady();
	client.dispose();
	assert.equal(client.state, "disposed");
	await assert.rejects(client.ensureReady(), /disposed/);
});

test("parseRpcMessage classifies JSON-RPC shapes", () => {
	assert.deepEqual(parseRpcMessage({ jsonrpc: "2.0", id: "a", method: "x", params: 1 }), {
		kind: "request",
		id: "a",
		method: "x",
		params: 1,
	});
	assert.equal(parseRpcMessage({ jsonrpc: "2.0", method: "n" })?.kind, "notification");
	assert.deepEqual(parseRpcMessage({ jsonrpc: "2.0", id: 1, result: null }), {
		kind: "response",
		id: 1,
		outcome: { ok: true, result: null },
	});
	assert.deepEqual(parseRpcMessage({ jsonrpc: "2.0", id: 2, error: { code: -32000, message: "auth" } }), {
		kind: "response",
		id: 2,
		outcome: { ok: false, error: { code: -32000, message: "auth" } },
	});
	assert.equal(parseRpcMessage({ jsonrpc: "2.0", id: 3 }), null);
	assert.equal(parseRpcMessage([1, 2]), null);
});
