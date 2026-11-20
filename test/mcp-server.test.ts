import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { request as httpRequest, type IncomingMessage } from "node:http";
import { mkdtempSync, readdirSync, rmSync } from "node:fs";
import { connect } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { PiMcpServer, type McpCall, type PiMcpServerOptions } from "../src/mcp-server.ts";

const servers: PiMcpServer[] = [];
test.afterEach(() => {
	for (const server of servers.splice(0)) server.dispose();
});

async function startServer(calls: McpCall[] = [], options: Partial<PiMcpServerOptions> = {}): Promise<PiMcpServer> {
	const server = new PiMcpServer({
		tools: () => [{ name: "echo", description: "Echoes", inputSchema: { type: "object" } }],
		call: async (call) => {
			calls.push(call);
			return { content: [{ type: "text", text: "ok" }] };
		},
		...options,
	});
	servers.push(server);
	await server.start();
	return server;
}

const listTools = { jsonrpc: "2.0", id: 1, method: "tools/list", params: {} };

async function post(url: string, token: string): Promise<number> {
	const response = await fetch(url, {
		method: "POST",
		headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
		body: JSON.stringify(listTools),
	});
	await response.arrayBuffer();
	return response.status;
}

test("http MCP answers only a minted, unrevoked nonce of the agent in the path, with that session's token", async () => {
	const server = await startServer();
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "http", nonce);
	assert.ok("url" in entry);
	const token = entry.headers[0]!.value.replace("Bearer ", "");
	const otherNonce = server.mint("a");
	const otherAgent = server.serverFor("b", "http", server.mint("b"));
	assert.ok("url" in otherAgent);
	assert.equal(await post(entry.url, token), 200);
	assert.equal(await post(entry.url, "0".repeat(token.length)), 401, "a wrong token");
	assert.equal(await post(entry.url, token.slice(0, -1)), 401, "a truncated token");
	assert.equal(await post(entry.url.replace(nonce, otherNonce), token), 401, "the token of another nonce of the same agent");
	assert.equal(await post(entry.url.replace("/mcp/a/", "/mcp/b/"), token), 401, "the token on another agent's path");
	assert.equal(await post(otherAgent.url, token), 401, "the token on another agent's session");
	assert.equal(await post(entry.url.replace(nonce, "0".repeat(32)), token), 401, "an unknown nonce");
	assert.equal(await post(entry.url.replace(`/${nonce}`, ""), token), 404, "no nonce");
	server.revoke(nonce);
	assert.equal(await post(entry.url, token), 403, "a revoked nonce");
});

/** Sends headers declaring `declared` bytes (or chunked when undefined), then `sent` bytes, without ending the request; waits for the server to close unless `waitClose` is false. */
async function oversized(url: string, headers: Record<string, string>, sent: number, declared?: number, waitClose = true) {
	const request = httpRequest(url, { method: "POST", headers: declared === undefined ? headers : { ...headers, "content-length": String(declared) } });
	request.on("error", () => {});
	const responded = once(request, "response") as Promise<[IncomingMessage]>;
	const closed = once(request, "close");
	if (sent > 0) request.write("x".repeat(sent));
	else request.flushHeaders();
	const timeout = setTimeout(() => request.destroy(new Error("no answer")), 2000);
	const [response] = await responded;
	response.resume();
	if (waitClose) await closed;
	else request.destroy();
	clearTimeout(timeout);
	return { status: response.statusCode, finished: request.writableFinished };
}

test("http MCP answers 413 to a declared oversized body without reading it, and closes the connection", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const endpoint = httpEndpoint(server);
	const reply = await oversized(endpoint.url, endpoint.headers, 0, 100 * 1024 * 1024);
	assert.equal(reply.status, 413);
	assert.equal(reply.finished, false, "answered before the client sent the body");
	assert.equal((await endpoint.send(listTools)).status, 200);
});

test("http MCP answers 413 once a chunked body passes the cap, and closes the connection", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const endpoint = httpEndpoint(server);
	const reply = await oversized(endpoint.url, endpoint.headers, 2048);
	assert.equal(reply.status, 413);
	assert.equal(reply.finished, false, "answered before the client ended the body");
});

test("http MCP refuses an oversized body only after authentication", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const endpoint = httpEndpoint(server);
	const reply = await oversized(endpoint.url, { ...endpoint.headers, authorization: "Bearer wrong" }, 0, 100 * 1024 * 1024, false);
	assert.equal(reply.status, 401);
});

/** The relay handshake of `agentId`'s session `nonce`, as the relay would send it. */
function helloFor(server: PiMcpServer, agentId: string, nonce: string) {
	const entry = server.serverFor(agentId, "stdio", nonce);
	assert.ok("env" in entry);
	const env = Object.fromEntries(entry.env.map((variable) => [variable.name, variable.value]));
	return { token: env.PI_MCP_TOKEN, agent: env.PI_MCP_AGENT, session: env.PI_MCP_SESSION };
}

/** Writes each of `writes` to the relay socket, 20 ms apart; reports whether the server closed the connection within 300 ms, and its replies. */
async function rawRelay(server: PiMcpServer, writes: string[]) {
	const entry = server.serverFor("a", "stdio", "unused");
	assert.ok("args" in entry);
	const socket = connect(entry.args[1]!);
	await once(socket, "connect");
	socket.on("error", () => {});
	const replies: Record<string, any>[] = [];
	createInterface({ input: socket }).on("line", (reply) => replies.push(JSON.parse(reply))).on("error", () => {});
	const closed = once(socket, "close").then(() => true);
	for (const data of writes) {
		socket.write(data);
		await new Promise((resolve) => setTimeout(resolve, 20));
	}
	const wasClosed = await Promise.race([closed, new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 300))]);
	socket.destroy();
	return { closed: wasClosed, replies };
}

const line = (message: unknown) => `${JSON.stringify(message)}\n`;
const paddedList = (bytes: number) => ({ ...listTools, params: { pad: "x".repeat(bytes) } });

test("stdio MCP refuses another agent's token and closes an oversized handshake line", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const nonce = server.mint("a");
	const otherHello = helloFor(server, "b", server.mint("b"));
	const handshake = (data: string) => rawRelay(server, [data]);
	const withCall = (hello: object) => line(hello) + line(listTools);
	const foreign = await handshake(withCall({ token: otherHello.token, agent: "a", session: nonce }));
	assert.equal(foreign.closed, true, "another agent's token");
	assert.deepEqual(foreign.replies, []);
	const huge = await handshake(`{"token":"${"x".repeat(8192)}`);
	assert.equal(huge.closed, true, "an unterminated handshake line over the cap");
	const good = await handshake(withCall(helloFor(server, "a", nonce)));
	assert.equal(good.replies.length, 1, "the session's own token");
	assert.equal(good.closed, false);
});

test("stdio MCP applies the request limit, not the handshake limit, to a request coalesced with the handshake", async () => {
	const server = await startServer([], { maxRequestBytes: 16 * 1024 });
	const hello = helloFor(server, "a", server.mint("a"));
	const request = JSON.stringify(paddedList(8 * 1024));
	const reply = await rawRelay(server, [line(hello) + request.slice(0, -10), `${request.slice(-10)}\n`]);
	assert.equal(reply.closed, false);
	assert.equal(reply.replies.length, 1);
	assert.ok(reply.replies[0]?.result, "the request over the handshake limit is answered");
});

test("stdio MCP closes the connection on a completed line over the cap without answering it", async () => {
	const server = await startServer([], { maxRequestBytes: 1024 });
	const hello = helloFor(server, "a", server.mint("a"));
	const inOneChunk = await rawRelay(server, [line(hello) + line(paddedList(4096))]);
	assert.equal(inOneChunk.closed, true, "a line that ends in the chunk it started in");
	assert.deepEqual(inOneChunk.replies, []);
	const big = line(paddedList(4096));
	const acrossChunks = await rawRelay(server, [line(hello), big.slice(0, 900), big.slice(900)]);
	assert.equal(acrossChunks.closed, true, "a line whose newline arrives in a later chunk");
	assert.deepEqual(acrossChunks.replies, []);
});

test("http MCP refuses a request whose nonce was revoked while its body uploaded", async () => {
	const server = await startServer();
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "http", nonce);
	assert.ok("url" in entry);
	const body = JSON.stringify(listTools);
	const request = httpRequest(entry.url, {
		method: "POST",
		headers: { ...Object.fromEntries(entry.headers.map((header) => [header.name, header.value])), "content-type": "application/json" },
	});
	const responded = once(request, "response") as Promise<[IncomingMessage]>;
	request.write(body.slice(0, 5));
	await new Promise((resolve) => setTimeout(resolve, 50));
	server.revoke(nonce);
	request.end(body.slice(5));
	const [response] = await responded;
	response.resume();
	assert.equal(response.statusCode, 403);
});

/** An HTTP endpoint for a freshly minted nonce of agent "a", with helpers that never hang a failing test. */
function httpEndpoint(server: PiMcpServer) {
	const nonce = server.mint("a");
	const entry = server.serverFor("a", "http", nonce);
	assert.ok("url" in entry);
	const headers: Record<string, string> = {
		...Object.fromEntries(entry.headers.map((header) => [header.name, header.value])),
		"content-type": "application/json",
	};
	const send = async (body: unknown, session?: string, method = "POST", url = entry.url) => {
		const response = await fetch(url, {
			method,
			headers: session ? { ...headers, "mcp-session-id": session } : headers,
			...(method === "POST" ? { body: JSON.stringify(body) } : {}),
			signal: AbortSignal.timeout(2000),
		});
		const text = await response.text();
		return { status: response.status, session: response.headers.get("mcp-session-id"), json: text ? (JSON.parse(text) as Record<string, any>) : null };
	};
	return { nonce, url: entry.url, headers, send };
}

const initialize = { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-06-18" } };
const callEcho = (id: number) => ({ jsonrpc: "2.0", id, method: "tools/call", params: { name: "echo", arguments: {} } });

/** Records unhandled rejections while `body` runs, so a crash of Pi's process shows up as a test failure. */
async function withoutUnhandledRejections(body: () => Promise<void>): Promise<void> {
	const rejections: unknown[] = [];
	const listener = (reason: unknown) => rejections.push(reason);
	process.on("unhandledRejection", listener);
	try {
		await body();
		await new Promise((resolve) => setTimeout(resolve, 50));
	} finally {
		process.off("unhandledRejection", listener);
	}
	assert.deepEqual(rejections, []);
}

/** A call handler that stays pending until its signal aborts; `aborted` resolves with the aborted call. */
function pendingCalls() {
	let started!: (call: McpCall) => void;
	const running = new Promise<McpCall>((resolve) => (started = resolve));
	const call = (mcpCall: McpCall) =>
		new Promise<never>((_resolve, reject) => {
			started(mcpCall);
			mcpCall.signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
		});
	return { call, running };
}

/** Calls with `{ quick: true }` succeed at once; others stay pending until aborted. `started` lists every call that began. */
function controlledCalls() {
	const started: McpCall[] = [];
	const waiters: (() => void)[] = [];
	const call = (mcpCall: McpCall) => {
		started.push(mcpCall);
		for (const waiter of waiters.splice(0)) waiter();
		if (mcpCall.arguments.quick === true) return Promise.resolve({ content: [{ type: "text" as const, text: "quick" }] });
		return new Promise<never>((_resolve, reject) => {
			mcpCall.signal.addEventListener("abort", () => reject(new Error("cancelled")), { once: true });
		});
	};
	const startedCount = async (count: number) => {
		while (started.length < count) await new Promise<void>((resolve) => waiters.push(resolve));
	};
	return { call, started, startedCount };
}

/** Starts a raw POST of `body`; `disconnect` drops it as a client that went away would. */
function rawPost(url: string, headers: Record<string, string>, body: unknown) {
	const request = httpRequest(url, { method: "POST", headers });
	request.on("error", () => {});
	const responded = once(request, "response").then(async (args) => {
		const response = args[0] as IncomingMessage;
		let text = "";
		for await (const chunk of response) text += chunk;
		return { status: response.statusCode, json: JSON.parse(text) as Record<string, any> };
	});
	responded.catch(() => {});
	request.end(JSON.stringify(body));
	return { responded, disconnect: () => request.destroy() };
}

const callWith = (id: number, args: Record<string, unknown>) => ({ jsonrpc: "2.0", id, method: "tools/call", params: { name: "echo", arguments: args } });

test("http MCP answers 400 with a JSON-RPC parse error to malformed JSON and keeps serving", async () => {
	const server = await startServer();
	const endpoint = httpEndpoint(server);
	await withoutUnhandledRejections(async () => {
		const response = await fetch(endpoint.url, { method: "POST", headers: endpoint.headers, body: "{\"jsonrpc\": \"2.0\", ", signal: AbortSignal.timeout(2000) });
		assert.equal(response.status, 400);
		assert.equal(((await response.json()) as Record<string, any>).error?.code, -32700);
	});
	assert.equal((await endpoint.send(listTools)).status, 200);
});

test("http MCP refuses a tools/call whose id is already running in the MCP session", { timeout: 5000 }, async () => {
	const calls = controlledCalls();
	const server = await startServer([], { call: calls.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const first = endpoint.send(callWith(7, {}), init.session!);
	await calls.startedCount(1);
	const duplicate = await endpoint.send(callWith(7, {}), init.session!);
	assert.equal(duplicate.json?.error?.code, -32600);
	assert.equal(calls.started.length, 1, "the duplicate never reaches Pi");
	assert.equal((await endpoint.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 7 } }, init.session!)).status, 202);
	assert.ok(calls.started[0]!.signal.aborted, "the first call is still the one cancelled by its id");
	assert.equal((await first).json?.result?.isError, true);
});

test("http MCP aborts only the calls of the POST whose client went away", { timeout: 5000 }, async () => {
	const calls = controlledCalls();
	const server = await startServer([], { call: calls.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const headers = { ...endpoint.headers, "mcp-session-id": init.session! };
	const batch = rawPost(endpoint.url, headers, [callWith(7, { quick: true }), callWith(8, {})]);
	await calls.startedCount(2);
	await new Promise((resolve) => setImmediate(resolve));
	const other = rawPost(endpoint.url, headers, callWith(7, {}));
	await calls.startedCount(3);
	batch.disconnect();
	await new Promise((resolve) => setTimeout(resolve, 50));
	assert.ok(calls.started[1]!.signal.aborted, "the gone POST's pending call");
	assert.equal(calls.started[2]!.signal.aborted, false, "another POST's call that reuses a finished id");
	other.disconnect();
	await new Promise((resolve) => setTimeout(resolve, 50));
	assert.ok(calls.started[2]!.signal.aborted);
});

test("http MCP answers 400 to a malformed agent path and keeps serving", async () => {
	const server = await startServer();
	const endpoint = httpEndpoint(server);
	await withoutUnhandledRejections(async () => {
		const malformed = await endpoint.send(listTools, undefined, "POST", endpoint.url.replace("/mcp/a/", "/mcp/%E0/"));
		assert.equal(malformed.status, 400);
	});
	assert.equal((await endpoint.send(listTools)).status, 200);
});

test("http MCP survives a client that aborts mid-upload", async () => {
	const server = await startServer();
	const endpoint = httpEndpoint(server);
	await withoutUnhandledRejections(async () => {
		const request = httpRequest(endpoint.url, { method: "POST", headers: { ...endpoint.headers, "content-length": "1000" } });
		request.on("error", () => {});
		request.write("{\"jsonrpc\":");
		await new Promise((resolve) => setTimeout(resolve, 50));
		request.destroy();
	});
	assert.equal((await endpoint.send(listTools)).status, 200);
});

test("http and stdio MCP answer a throwing handler with a JSON-RPC error", async () => {
	const server = await startServer([], {
		tools: () => {
			throw new Error("boom");
		},
	});
	const endpoint = httpEndpoint(server);
	await withoutUnhandledRejections(async () => {
		const reply = await endpoint.send(listTools);
		assert.equal(reply.status, 200);
		assert.equal(reply.json?.error?.code, -32603);
		const stdio = await relay(server, "a", endpoint.nonce, endpoint.nonce);
		assert.equal((stdio.replies[0] as Record<string, any> | undefined)?.error?.code, -32603);
		stdio.socket.destroy();
	});
});

test("http MCP cancels a call from a separate POST of the same MCP session", async () => {
	const pending = pendingCalls();
	const server = await startServer([], { call: pending.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	assert.ok(init.session, "initialize issues an Mcp-Session-Id");
	const reply = endpoint.send(callEcho(7), init.session!);
	const call = await pending.running;
	assert.equal((await endpoint.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: 7 } }, init.session!)).status, 202);
	assert.ok(call.signal.aborted);
	assert.equal((await reply).json?.result?.isError, true);
});

test("http MCP aborts a call in flight when its nonce is revoked", async () => {
	const pending = pendingCalls();
	const server = await startServer([], { call: pending.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const reply = endpoint.send(callEcho(7), init.session!);
	const call = await pending.running;
	server.revoke(endpoint.nonce);
	assert.ok(call.signal.aborted);
	assert.equal((await reply).json?.result?.isError, true);
	assert.equal((await endpoint.send(listTools, init.session!)).status, 403);
});

test("http MCP DELETE aborts the MCP session's calls and ends the session", async () => {
	const pending = pendingCalls();
	const server = await startServer([], { call: pending.call });
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const reply = endpoint.send(callEcho(7), init.session!);
	const call = await pending.running;
	assert.equal((await endpoint.send(null, init.session!, "DELETE")).status, 200);
	assert.ok(call.signal.aborted);
	assert.equal((await reply).json?.result?.isError, true);
	assert.equal((await endpoint.send(listTools, init.session!)).status, 404, "an ended MCP session");
	assert.equal((await endpoint.send(listTools)).status, 200, "a request without a session");
});

test("http MCP answers 404 to a request whose MCP session was deleted while its body uploaded", async () => {
	const server = await startServer();
	const endpoint = httpEndpoint(server);
	const init = await endpoint.send(initialize);
	const body = JSON.stringify(listTools);
	const request = httpRequest(endpoint.url, { method: "POST", headers: { ...endpoint.headers, "mcp-session-id": init.session! } });
	const responded = once(request, "response") as Promise<[IncomingMessage]>;
	request.write(body.slice(0, 5));
	await new Promise((resolve) => setTimeout(resolve, 50));
	assert.equal((await endpoint.send(null, init.session!, "DELETE")).status, 200);
	request.end(body.slice(5));
	const [response] = await responded;
	response.resume();
	assert.equal(response.statusCode, 404);
});

/** Opens a raw stdio-relay connection and sends its handshake; resolves once the server answered or closed. */
async function relay(server: PiMcpServer, agentId: string, nonce: string, session: string) {
	const entry = server.serverFor(agentId, "stdio", nonce);
	assert.ok("args" in entry);
	const env = Object.fromEntries(entry.env.map((variable) => [variable.name, variable.value]));
	assert.equal(env.PI_MCP_SESSION, nonce);
	const socket = connect(entry.args[1]!);
	await once(socket, "connect");
	const closed = once(socket, "close");
	const lines = createInterface({ input: socket });
	lines.on("error", () => {});
	const replies: unknown[] = [];
	lines.on("line", (line) => replies.push(JSON.parse(line)));
	socket.write(`${JSON.stringify({ token: env.PI_MCP_TOKEN, agent: env.PI_MCP_AGENT, session })}\n`);
	socket.write(`${JSON.stringify(listTools)}\n`);
	const answered = new Promise<void>((resolve) => lines.once("line", () => resolve()));
	await Promise.race([answered, closed]);
	return { socket, closed, replies };
}

test("stdio MCP refuses a wrong nonce and closes a connection once its nonce is revoked", async () => {
	const server = await startServer();
	const nonce = server.mint("a");
	const wrong = await relay(server, "a", nonce, "f".repeat(32));
	await wrong.closed;
	assert.deepEqual(wrong.replies, []);

	const good = await relay(server, "a", nonce, nonce);
	assert.equal(good.replies.length, 1);
	server.revoke(nonce);
	await good.closed;
	assert.ok(good.socket.destroyed);
});

const serverHandles = () => process.getActiveResourcesInfo().filter((name) => name === "TCPServerWrap" || name === "PipeServerWrap").length;

test("disposing a pending start rejects it and leaves no listener or socket directory", { skip: process.platform === "win32" }, async () => {
	const dir = mkdtempSync(join(tmpdir(), "pi-acp-dispose-"));
	const saved = process.env.TMPDIR;
	process.env.TMPDIR = dir;
	try {
		// WHY: an earlier test's disposed servers may still be closing.
		await new Promise((resolve) => setTimeout(resolve, 50));
		const before = serverHandles();
		const server = new PiMcpServer({ tools: () => [], call: async () => ({ content: [] }) });
		const starting = server.start();
		server.dispose();
		const outcome = await Promise.race([
			starting.then(() => "started", (error: unknown) => (error instanceof Error ? error.message : String(error))),
			new Promise<string>((resolve) => setTimeout(() => resolve("still pending"), 2000)),
		]);
		assert.match(outcome, /disposed while starting/);
		await new Promise((resolve) => setTimeout(resolve, 50));
		assert.equal(serverHandles(), before, "no transport is left listening");
		assert.deepEqual(readdirSync(dir), [], "no socket directory is left behind");
	} finally {
		if (saved === undefined) delete process.env.TMPDIR;
		else process.env.TMPDIR = saved;
		rmSync(dir, { recursive: true, force: true });
	}
});
