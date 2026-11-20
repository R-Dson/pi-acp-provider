import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
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

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

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
	assert.equal(await client.ensureSession(), first, "an unused session in the same cwd is kept");
	await runPrompt(client, "use it");
	client.resetSession(process.cwd());
	assert.notEqual(await client.ensureSession(), first, "a used session is replaced");
});

test("each conversation has its own session, reset on its own", async () => {
	const client = makeClient();
	const [a, b] = await Promise.all([client.ensureSession({ conversation: "a" }), client.ensureSession({ conversation: "b" })]);
	assert.notEqual(a, b);
	for (const sessionId of [a, b]) await client.prompt({ sessionId, payload: promptPayload(sessionId, "use it"), onUpdate: () => {} });
	client.resetSession(process.cwd(), "a");
	assert.notEqual(await client.ensureSession({ conversation: "a" }), a, "the reset conversation gets a new session");
	assert.equal(await client.ensureSession({ conversation: "b" }), b, "the other conversation keeps its session");
});

test("spec-shaped session updates stream in order and the turn finishes", async () => {
	const client = makeClient();
	const { result, updates, reply } = await runPrompt(client, "do the thing");
	assert.deepEqual(result, { kind: "finished", stopReason: "end_turn", usage: undefined });
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

/** Starts a SLOW prompt on the current session and aborts it at its first update. */
async function cancelSlowPrompt(client: AcpClient, text = "SLOW work"): Promise<string> {
	const sessionId = await client.ensureSession();
	const controller = new AbortController();
	const result = await client.prompt({
		sessionId,
		payload: promptPayload(sessionId, text),
		signal: controller.signal,
		onUpdate: () => controller.abort(),
	});
	assert.deepEqual(result, { kind: "aborted" });
	return sessionId;
}

test("a cancel that never drains moves the next prompt to a new session; a second one restarts the agent", async () => {
	const ended: string[] = [];
	const asked: string[] = [];
	const client = makeClient({
		agent: { env: { FAKE_IGNORE_CANCEL: "1" } },
		cancelGraceMs: 200,
		onSessionEnd: (sessionId) => ended.push(sessionId),
		onServerRequest: async (request) => {
			asked.push(String(request.sessionId));
			return approveAll(request);
		},
	});
	const firstPid = (await client.ensureReady()).name;
	const first = await cancelSlowPrompt(client);
	const second = await client.ensureSession();
	assert.notEqual(second, first);
	assert.equal(client.info?.name, firstPid, "one timeout keeps the process");
	assert.deepEqual(ended, [first]);
	const echo = await runPrompt(client, "ECHO two");
	assert.equal(echo.result.kind, "finished");
	assert.match(echo.reply, new RegExp(`^ECHO\\[${second}\\]<<<ECHO two>>> STALE<<<permission=cancelled;mcp=none>>>$`));
	assert.deepEqual(asked, [], "the ended session's permission request never reaches the handler");

	assert.equal(await cancelSlowPrompt(client), second);
	await client.ensureSession();
	assert.notEqual(client.info?.name, firstPid, "a second timeout in a row restarts the process");
	assert.deepEqual(ended, [first, second]);
	const after = await runPrompt(client, "ECHO three");
	assert.equal(after.result.kind, "finished");
});

test("a new session that fails after a drain timeout restarts the agent", async () => {
	const client = makeClient({ agent: { env: { FAKE_IGNORE_CANCEL: "1", FAKE_FAIL_SESSION_N: "2" } }, cancelGraceMs: 200 });
	const firstPid = (await client.ensureReady()).name;
	await cancelSlowPrompt(client);
	await client.ensureSession();
	assert.notEqual(client.info?.name, firstPid);
	const { result } = await runPrompt(client, "ECHO again");
	assert.equal(result.kind, "finished");
});

test("a replacement session that fails after its caller aborted still restarts the agent", async () => {
	const client = makeClient({
		agent: { env: { FAKE_IGNORE_CANCEL: "1", FAKE_FAIL_SESSION_N: "2", FAKE_FAIL_SESSION_DELAY: "500" } },
		cancelGraceMs: 200,
	});
	const firstPid = (await client.ensureReady()).name;
	await cancelSlowPrompt(client);
	const controller = new AbortController();
	const waiting = client.ensureSession({ signal: controller.signal });
	// After the 200 ms drain timeout, while the replacement session/new is still pending.
	await sleep(350);
	controller.abort();
	await assert.rejects(waiting, /aborted/);
	await sleep(400);
	await client.ensureSession();
	assert.notEqual(client.info?.name, firstPid, "the failed replacement restarted the agent");
});

test("a cancel that drains in time breaks the drain-timeout streak", async () => {
	const client = makeClient({ cancelGraceMs: 200 });
	const firstPid = (await client.ensureReady()).name;
	const first = await cancelSlowPrompt(client, "SLOW IGNORE-CANCEL");
	const second = await cancelSlowPrompt(client);
	assert.notEqual(second, first);
	// The timely drain settles before anything waits on it.
	await sleep(100);
	assert.equal(await cancelSlowPrompt(client, "SLOW IGNORE-CANCEL"), second);
	const third = await client.ensureSession();
	assert.notEqual(third, second, "the second ignored cancel gets a new session");
	assert.equal(client.info?.name, firstPid, "not a restart: a timely drain came between the timeouts");
});

test("ending a session while its cancel drains rejects the cancelled prompt", async () => {
	const client = makeClient({ agent: { env: { FAKE_IGNORE_CANCEL: "1" } }, cancelGraceMs: 200 });
	await cancelSlowPrompt(client);
	client.resetSession(process.cwd());
	await client.ensureSession();
	// WHY: white-box, because nothing else observes a request the agent never answers.
	const pending = (client as unknown as { pending: Map<unknown, unknown> }).pending;
	assert.equal(pending.size, 0, "the cancelled session/prompt is no longer pending");
	const { result } = await runPrompt(client, "ECHO after reset");
	assert.equal(result.kind, "finished");
});

/** A server-request handler that holds each request until the test answers it. */
function heldHandler() {
	const held: ((optionId: string) => void)[] = [];
	const arrived: (() => void)[] = [];
	const handler: ServerRequestHandler = () =>
		new Promise((resolve) => {
			held.push((optionId) => resolve({ outcome: { outcome: "selected", optionId } }));
			arrived.shift()?.();
		});
	const next = async (): Promise<(optionId: string) => void> => {
		while (held.length === 0) await new Promise<void>((resolve) => arrived.push(resolve));
		return held.shift()!;
	};
	return { handler, next };
}

/** Starts a PERMISSION prompt that ignores its cancel, waits for its request to be held, then aborts it. */
async function cancelWhileAsking(client: AcpClient, next: () => Promise<(optionId: string) => void>) {
	const sessionId = await client.ensureSession();
	const controller = new AbortController();
	const prompt = client.prompt({
		sessionId,
		payload: promptPayload(sessionId, "PERMISSION IGNORE-CANCEL"),
		signal: controller.signal,
		onUpdate: () => {},
	});
	const answer = await next();
	controller.abort();
	assert.deepEqual(await prompt, { kind: "aborted" });
	return answer;
}

test("a permission answer for a session that ended meanwhile goes back as cancelled", async () => {
	const { handler, next } = heldHandler();
	const client = makeClient({ cancelGraceMs: 200, onServerRequest: handler });
	const answer = await cancelWhileAsking(client, next);
	await client.ensureSession();
	answer("allow");
	await sleep(50);
	const { reply } = await runPrompt(client, "ECHO after");
	assert.match(reply, /CANCELLEDPERM<<<cancelled>>>/);
});

test("a permission answer never reaches the agent process that replaced its own", async () => {
	const { handler, next } = heldHandler();
	const client = makeClient({ cancelGraceMs: 200, onServerRequest: handler });
	const firstPid = (await client.ensureReady()).name;
	await cancelSlowPrompt(client, "SLOW IGNORE-CANCEL");
	const stale = await cancelWhileAsking(client, next);
	await client.ensureSession();
	assert.notEqual(client.info?.name, firstPid, "the second drain timeout restarted the agent");
	// The old process's held request was its second (srv-2, after the stale one); the new process's
	// second request reuses that id.
	const first = runPrompt(client, "PERMISSION");
	(await next())("allow");
	await first;
	const asking = runPrompt(client, "PERMISSION");
	const current = await next();
	stale("allow");
	await sleep(50);
	current("deny");
	assert.match((await asking).reply, /\[approved=deny\]/);
});

test("permission requests without a string session id are answered cancelled", async () => {
	const asked: unknown[] = [];
	const client = makeClient({
		onServerRequest: async (request) => {
			asked.push(request.params);
			return approveAll(request);
		},
	});
	const { reply } = await runPrompt(client, "BADSESSIONPERM");
	assert.equal(reply, "BADPERM<<<missing=cancelled;number=cancelled>>>");
	assert.deepEqual(asked, []);
});

const isAlive = (pid: number): boolean => {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
};

test("a restart kills process-group members that outlive the agent", { skip: process.platform === "win32" }, async () => {
	const pidFile = join(mkdtempSync(join(tmpdir(), "fake-orphan-")), "pid");
	const readPid = async (): Promise<number> => {
		for (let tries = 0; tries < 100 && !existsSync(pidFile); tries++) await sleep(20);
		return Number(readFileSync(pidFile, "utf8"));
	};
	const orphans: number[] = [];
	try {
		const client = makeClient({ agent: { env: { FAKE_IGNORE_CANCEL: "1", FAKE_ORPHAN: pidFile } }, cancelGraceMs: 200 });
		const firstAgent = (await client.ensureReady()).name;
		const orphan = await readPid();
		orphans.push(orphan);
		rmSync(pidFile);
		await cancelSlowPrompt(client);
		await cancelSlowPrompt(client);
		await client.ensureSession();
		assert.notEqual(client.info?.name, firstAgent, "the second drain timeout restarted the agent");
		orphans.push(await readPid());
		// The agent's SIGKILL follows its SIGTERM after a 3 s grace.
		for (let tries = 0; tries < 50 && isAlive(orphan); tries++) await sleep(100);
		assert.equal(isAlive(orphan), false, "the old agent's SIGTERM-ignoring child is killed");
	} finally {
		for (const pid of orphans) if (isAlive(pid)) process.kill(pid, "SIGKILL");
		rmSync(dirname(pidFile), { recursive: true, force: true });
	}
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

test("an agent answering with an unsupported protocol version is rejected", async () => {
	const client = makeClient({ agent: { env: { FAKE_PROTOCOL: "2" } } });
	await assert.rejects(client.ensureReady(), /speaks ACP v2; pi-acp-provider supports v1/);
	assert.equal(client.state, "uninitialized");
});

test("a parent SIGTERM kills every agent and still terminates the host", async () => {
	// Two clients must not each think the other handles the signal.
	const clientUrl = new URL("../src/acp-client.ts", import.meta.url).href;
	const script = `
		import { AcpClient } from ${JSON.stringify(clientUrl)};
		const agent = { name: "fake", command: process.execPath, args: [${JSON.stringify(fakePath)}], env: {}, authMethod: undefined };
		const clients = [1, 2].map(() => new AcpClient({ agent, cwd: process.cwd(), handshakeTimeoutMs: 8000, onServerRequest: async () => null }));
		const infos = await Promise.all(clients.map((client) => client.ensureReady()));
		console.log(infos.map((info) => info.name.replace("fake-agent-", "")).join(" "));
		setInterval(() => {}, 1000);
		process.kill(process.pid, "SIGTERM");
	`;
	const host = spawn(process.execPath, ["--input-type=module", "-e", script], { stdio: ["ignore", "pipe", "inherit"] });
	let stdout = "";
	host.stdout.on("data", (chunk: Buffer) => (stdout += chunk));
	const watchdog = setTimeout(() => host.kill("SIGKILL"), 8000);
	const [, signal] = await once(host, "exit");
	clearTimeout(watchdog);
	assert.equal(signal, "SIGTERM");

	const alive = (pid: number) => {
		try {
			process.kill(pid, 0);
			return true;
		} catch {
			return false;
		}
	};
	const pids = stdout.trim().split(" ").map(Number);
	assert.equal(pids.length, 2);
	for (let tries = 0; tries < 50 && pids.some(alive); tries++) await new Promise((resolve) => setTimeout(resolve, 20));
	assert.deepEqual(pids.filter(alive), [], "agent processes are gone");
});
