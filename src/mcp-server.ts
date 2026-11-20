/**
 * MCP server that offers Pi's tools to ACP agents.
 *
 * One server per provider serves every agent. Each ACP session gets its own
 * revocable nonce, minted by `mint`; the agent id and the nonce travel in the
 * HTTP path or the stdio relay's handshake, so a call names the session it
 * came from, and a revoked session's calls are refused.
 * Tool lists are read live, so a changed Pi loadout is visible on the next
 * `tools/list`; stdio connections are also told with `list_changed`.
 *
 * Two transports, both bound to this machine. Each session's token is an HMAC of
 * its agent id and nonce under a per-process key, compared in constant time:
 * - HTTP (MCP Streamable HTTP, JSON responses only) on 127.0.0.1.
 * - stdio: the agent spawns `mcp-relay.mjs`, which pipes its stdin/stdout to a
 *   unix socket (a named pipe on Windows) in a private temp directory. Every
 *   ACP agent must support stdio MCP servers, so this is the baseline.
 *
 * Only the MCP subset Pi needs is implemented: initialize, ping, tools/list,
 * tools/call and request cancellation.
 */

import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer as createHttpServer, type IncomingMessage, type Server as HttpServer, type ServerResponse } from "node:http";
import { createServer as createNetServer, type Server as NetServer, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { isRecord } from "./acp.ts";
import { PI_MCP_SERVER, type McpTransport } from "./config.ts";

const MCP_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"];
const RELAY_PATH = fileURLToPath(new URL("./mcp-relay.mjs", import.meta.url));
const IS_WINDOWS = process.platform === "win32";
const MAX_REQUEST_BYTES = 8 * 1024 * 1024;
/** The relay's handshake line is a few hundred bytes; this bounds what an unauthenticated peer can make Pi buffer. */
const MAX_HELLO_BYTES = 4096;

export interface McpTool {
	name: string;
	description: string;
	inputSchema: Record<string, unknown>;
}

export type McpContent = { type: "text"; text: string } | { type: "image"; data: string; mimeType: string };

export interface McpCallResult {
	content: McpContent[];
	isError?: boolean;
}

export interface McpCall {
	agentId: string;
	/** The session nonce the call came in on; still minted (not revoked) when the call arrived. */
	nonce: string;
	name: string;
	arguments: Record<string, unknown>;
	/** Aborts when the agent cancels the request or disconnects. */
	signal: AbortSignal;
}

export interface PiMcpServerOptions {
	/** The tools offered on `agentId`'s session `nonce`. */
	tools: (agentId: string, nonce: string) => McpTool[];
	call: (call: McpCall) => Promise<McpCallResult>;
	log?: ((message: string) => void) | undefined;
	/** The largest HTTP body or stdio line accepted, in bytes; larger ones get 413 or a closed connection. */
	maxRequestBytes?: number;
}

/** An ACP `McpServer` entry for `session/new`. */
export type AcpMcpServer =
	| { name: string; command: string; args: string[]; env: { name: string; value: string }[] }
	| { type: "http"; name: string; url: string; headers: { name: string; value: string }[] };

type JsonRpcId = string | number;

const ROOT_COMBINATORS = ["anyOf", "oneOf", "allOf"] as const;

/**
 * Some agents (Claude Code) drop a tool whose schema has a top-level
 * combinator. Flatten it into one object, keeping the root properties first,
 * and state the lost constraint in the description; Pi still validates the
 * arguments against the full schema.
 */
export function flattenRootCombinators(schema: Record<string, unknown>): { schema: Record<string, unknown>; note?: string } {
	const present = ROOT_COMBINATORS.filter((key) => Array.isArray(schema[key]));
	if (present.length === 0) return { schema };
	const { anyOf: _anyOf, oneOf: _oneOf, allOf: _allOf, ...rest } = schema;
	const properties: Record<string, unknown> = isRecord(schema.properties) ? { ...schema.properties } : {};
	const required = new Set(Array.isArray(schema.required) ? schema.required.filter((key) => typeof key === "string") : []);
	const groups: string[] = [];
	for (const key of present) {
		for (const branch of schema[key] as unknown[]) {
			if (!isRecord(branch)) continue;
			const branchProps = isRecord(branch.properties) ? branch.properties : {};
			for (const [name, value] of Object.entries(branchProps)) if (!(name in properties)) properties[name] = value;
			const branchRequired = Array.isArray(branch.required) ? branch.required.filter((name) => typeof name === "string") : [];
			if (key === "allOf") for (const name of branchRequired) required.add(name);
			else groups.push((branchRequired.length > 0 ? branchRequired : Object.keys(branchProps)).join(", "));
		}
	}
	const flat: Record<string, unknown> = { ...rest, type: "object", properties };
	if (required.size > 0) flat.required = [...required];
	else delete flat.required;
	const exclusive = present.includes("oneOf");
	const note =
		groups.length > 0
			? `Input constraint: provide parameters for ${exclusive ? "exactly one" : "at least one"} of: ${groups.map((group) => `(${group})`).join(" or ")}.`
			: present.includes("allOf")
				? "Input constraint: all listed parameters apply together."
				: undefined;
	return note ? { schema: flat, note } : { schema: flat };
}

/** A Pi tool as an MCP tool; null when its parameters are not an object schema, which MCP requires. */
export function toMcpTool(tool: { name: string; description: string; parameters: unknown }): McpTool | null {
	// The JSON round trip drops TypeBox's symbol-keyed metadata.
	const parameters: unknown = JSON.parse(JSON.stringify(tool.parameters ?? {}));
	if (!isRecord(parameters)) return null;
	const { schema, note } = flattenRootCombinators(parameters);
	if (schema.type !== "object") return null;
	return { name: tool.name, description: [note, tool.description].filter(Boolean).join("\n\n"), inputSchema: schema };
}

/** Compares two secrets in constant time; hashing first hides their lengths too. */
function secretsEqual(given: string, expected: string): boolean {
	const digest = (value: string) => createHash("sha256").update(value).digest();
	return timingSafeEqual(digest(given), digest(expected));
}

/** The request body, or null once it exceeds `limit` bytes; the rest is left unread. Rejects when the client aborts. */
function readBody(request: IncomingMessage, limit: number): Promise<Buffer | null> {
	const declared = Number(request.headers["content-length"]);
	if (Number.isFinite(declared) && declared > limit) return Promise.resolve(null);
	return new Promise((resolve, reject) => {
		const chunks: Buffer[] = [];
		let size = 0;
		const stop = () => {
			request.off("data", onData).off("end", onEnd).off("error", onError).off("close", onClose);
			// WARN: an error after we stopped listening would otherwise be an uncaught 'error' event.
			request.on("error", () => {});
		};
		const onData = (chunk: Buffer) => {
			size += chunk.length;
			if (size > limit) {
				stop();
				request.pause();
				resolve(null);
			} else chunks.push(chunk);
		};
		const onEnd = () => {
			stop();
			resolve(Buffer.concat(chunks));
		};
		const onError = (error: Error) => {
			stop();
			reject(error);
		};
		const onClose = () => onError(new Error("aborted"));
		request.on("data", onData).on("end", onEnd).on("error", onError).on("close", onClose);
	});
}

/**
 * Calls `onLine` for each newline-terminated line of `socket`, in order and synchronously, so a line can change
 * the limit of the next. Destroys the socket once a line, complete or not, exceeds `limit()` bytes; an oversized
 * line is never decoded or parsed.
 */
function readLines(socket: Socket, limit: () => number, onLine: (line: string) => void): void {
	let pending: Buffer[] = [];
	let pendingBytes = 0;
	socket.on("data", (chunk: Buffer) => {
		let start = 0;
		while (!socket.destroyed) {
			const newline = chunk.indexOf(0x0a, start);
			const end = newline === -1 ? chunk.length : newline;
			pendingBytes += end - start;
			if (pendingBytes > limit()) return void socket.destroy();
			pending.push(chunk.subarray(start, end));
			if (newline === -1) return;
			const line = Buffer.concat(pending).toString("utf8");
			pending = [];
			pendingBytes = 0;
			start = newline + 1;
			onLine(line);
		}
	});
}

/** The decoded path segment, or null when it is not valid percent-encoding. */
function decodePathSegment(segment: string): string | null {
	try {
		return decodeURIComponent(segment);
	} catch {
		return null;
	}
}

const rpcResult = (id: JsonRpcId, result: unknown) => ({ jsonrpc: "2.0", id, result });
const rpcError = (id: JsonRpcId | null, code: number, message: string) => ({ jsonrpc: "2.0", id, error: { code, message } });

/** In-flight `tools/call` requests of one connection or HTTP MCP session, for `notifications/cancelled`. */
type Inflight = Map<JsonRpcId, AbortController>;

/** An HTTP MCP session, issued as `Mcp-Session-Id` on `initialize`, so a cancellation in a later POST reaches its call. */
interface HttpSession {
	nonce: string;
	inflight: Inflight;
}

const abortAll = (inflight: Inflight): void => {
	for (const controller of inflight.values()) controller.abort();
	inflight.clear();
};

export class PiMcpServer {
	private readonly options: PiMcpServerOptions;
	/** Derives each session's token, so a token is valid only for the agent and nonce it was issued with. */
	private readonly tokenKey = randomBytes(32);
	private readonly maxRequestBytes: number;
	private http: HttpServer | null = null;
	private httpPort = 0;
	private socketServer: NetServer | null = null;
	private socketDir: string | null = null;
	private socketPath = "";
	/** Authenticated stdio connections, with the agent and session nonce they named. */
	private readonly sockets = new Map<Socket, { agentId: string; nonce: string }>();
	/** Minted session nonces, by nonce, with the agent each was minted for. */
	private readonly nonces = new Map<string, string>();
	/**
	 * HTTP MCP sessions by `Mcp-Session-Id`; a request without that header shares its
	 * nonce's entry under the key `nonce:<nonce>`, which never collides with a minted id.
	 */
	private readonly httpSessions = new Map<string, HttpSession>();
	private started: Promise<void> | null = null;
	/** Bumped by `dispose`, so a start it interrupted neither succeeds nor closes a later start's transports. */
	private generation = 0;

	constructor(options: PiMcpServerOptions) {
		this.options = options;
		this.maxRequestBytes = options.maxRequestBytes ?? MAX_REQUEST_BYTES;
	}

	/**
	 * Starts both transports once; later calls wait for the first. A failed start is retried by the next
	 * call; a start that `dispose` interrupted rejects.
	 */
	start(): Promise<void> {
		if (this.started) return this.started;
		const generation = this.generation;
		const begin = (transport: () => Promise<void>): Promise<void> => {
			try {
				return transport();
			} catch (error) {
				return Promise.reject(error);
			}
		};
		const started = Promise.allSettled([begin(() => this.startHttp()), begin(() => this.startSocket())]).then((results) => {
			if (generation !== this.generation) throw new Error("Pi's MCP server was disposed while starting");
			const failed = results.find((result) => result.status === "rejected");
			if (!failed) return;
			// WHY: a transport that did come up would otherwise keep listening, unreachable, beside the retry's.
			this.closeTransports();
			this.started = null;
			throw failed.reason;
		});
		this.started = started;
		return started;
	}

	/** The bearer token for `agentId`'s session `nonce`; JSON keeps the two fields from running into each other. */
	private tokenFor(agentId: string, nonce: string): string {
		return createHmac("sha256", this.tokenKey).update(JSON.stringify([agentId, nonce])).digest("hex");
	}

	/** A new session nonce for `agentId`, valid until `revoke`. */
	mint(agentId: string): string {
		const nonce = randomBytes(16).toString("hex");
		this.nonces.set(nonce, agentId);
		return nonce;
	}

	/** Refuses the nonce from now on and closes the stdio connections that use it. */
	revoke(nonce: string): void {
		this.nonces.delete(nonce);
		for (const [key, session] of this.httpSessions) {
			if (session.nonce !== nonce) continue;
			this.httpSessions.delete(key);
			abortAll(session.inflight);
		}
		for (const [socket, owner] of this.sockets) {
			if (owner.nonce !== nonce) continue;
			this.sockets.delete(socket);
			socket.destroy();
		}
	}

	/** The `session/new` entry that connects `agentId`'s session `nonce` over `transport`. */
	serverFor(agentId: string, transport: McpTransport, nonce: string): AcpMcpServer {
		if (transport === "http") {
			return {
				type: "http",
				name: PI_MCP_SERVER,
				url: `http://127.0.0.1:${this.httpPort}/mcp/${encodeURIComponent(agentId)}/${nonce}`,
				headers: [{ name: "Authorization", value: `Bearer ${this.tokenFor(agentId, nonce)}` }],
			};
		}
		return {
			name: PI_MCP_SERVER,
			command: process.execPath,
			args: [RELAY_PATH, this.socketPath],
			// The token goes in the environment, which other users cannot read, not in argv.
			env: [
				{ name: "PI_MCP_TOKEN", value: this.tokenFor(agentId, nonce) },
				{ name: "PI_MCP_AGENT", value: agentId },
				{ name: "PI_MCP_SESSION", value: nonce },
			],
		};
	}

	/** Tells `agentId`'s stdio connections that its tool list changed. */
	notifyToolsChanged(agentId: string): void {
		for (const [socket, owner] of this.sockets) {
			if (owner.agentId === agentId) socket.write(`${JSON.stringify({ jsonrpc: "2.0", method: "notifications/tools/list_changed" })}\n`);
		}
	}

	dispose(): void {
		this.generation++;
		this.closeTransports();
		this.nonces.clear();
		this.started = null;
	}

	private closeTransports(): void {
		this.http?.close();
		this.http?.closeAllConnections();
		this.socketServer?.close();
		for (const socket of this.sockets.keys()) socket.destroy();
		this.sockets.clear();
		for (const session of this.httpSessions.values()) abortAll(session.inflight);
		this.httpSessions.clear();
		if (this.socketDir) rmSync(this.socketDir, { recursive: true, force: true });
		this.http = null;
		this.socketServer = null;
		this.socketDir = null;
	}

	/**
	 * Answers one JSON-RPC message; null for notifications and responses. Never rejects.
	 * A `tools/call` it starts adds its controller to `owned`, so the caller can abort exactly its own calls.
	 */
	private async handle(message: unknown, agentId: string, nonce: string, inflight: Inflight, owned?: Set<AbortController>): Promise<object | null> {
		try {
			return await this.dispatch(message, agentId, nonce, inflight, owned);
		} catch (error) {
			const id = isRecord(message) && (typeof message.id === "string" || typeof message.id === "number") ? message.id : null;
			this.options.log?.(`mcp request failed: ${error instanceof Error ? error.message : String(error)}`);
			return id === null ? null : rpcError(id, -32603, "Internal error");
		}
	}

	private async dispatch(
		message: unknown,
		agentId: string,
		nonce: string,
		inflight: Inflight,
		owned: Set<AbortController> | undefined,
	): Promise<object | null> {
		if (!isRecord(message) || typeof message.method !== "string") return null;
		const { method } = message;
		const id = typeof message.id === "string" || typeof message.id === "number" ? message.id : undefined;
		const params = isRecord(message.params) ? message.params : {};
		if (id === undefined) {
			if (method === "notifications/cancelled") {
				const requestId = params.requestId;
				if (typeof requestId === "string" || typeof requestId === "number") inflight.get(requestId)?.abort();
			}
			return null;
		}
		switch (method) {
			case "initialize": {
				const requested = typeof params.protocolVersion === "string" ? params.protocolVersion : "";
				return rpcResult(id, {
					protocolVersion: MCP_PROTOCOL_VERSIONS.includes(requested) ? requested : MCP_PROTOCOL_VERSIONS[0],
					capabilities: { tools: { listChanged: true } },
					serverInfo: { name: PI_MCP_SERVER, version: "1.0.0" },
					instructions: "Pi's tools. Pi runs each call itself, with its own approvals and display.",
				});
			}
			case "ping":
				return rpcResult(id, {});
			case "tools/list":
				return rpcResult(id, { tools: this.options.tools(agentId, nonce) });
			case "tools/call": {
				const name = typeof params.name === "string" ? params.name : "";
				const args = isRecord(params.arguments) ? params.arguments : {};
				// WHY: replacing a live id's controller would leave that call uncancellable and let another request abort this one.
				if (inflight.has(id)) return rpcError(id, -32600, `Request id ${JSON.stringify(id)} is already in use`);
				const controller = new AbortController();
				inflight.set(id, controller);
				owned?.add(controller);
				try {
					return rpcResult(id, await this.options.call({ agentId, nonce, name, arguments: args, signal: controller.signal }));
				} catch (error) {
					const text = error instanceof Error ? error.message : String(error);
					return rpcResult(id, { content: [{ type: "text", text }], isError: true });
				} finally {
					inflight.delete(id);
					owned?.delete(controller);
				}
			}
			default:
				return rpcError(id, -32601, `Method not found: ${method}`);
		}
	}

	private startHttp(): Promise<void> {
		const server = createHttpServer((request, response) => {
			this.serveHttp(request, response).catch((error: unknown) => {
				this.options.log?.(`mcp http request failed: ${error instanceof Error ? error.message : String(error)}`);
				if (!response.headersSent) response.writeHead(500).end();
				else response.destroy();
			});
		});
		// A tool call stays open while Pi runs the tool, which can wait on the user.
		server.requestTimeout = 0;
		server.timeout = 0;
		this.http = server;
		return new Promise((resolve, reject) => {
			server.once("error", reject);
			server.once("close", () => reject(new Error("Pi's MCP server was closed while starting")));
			server.listen(0, "127.0.0.1", () => {
				const address = server.address();
				this.httpPort = typeof address === "object" && address ? address.port : 0;
				resolve();
			});
		});
	}

	private async serveHttp(request: IncomingMessage, response: ServerResponse): Promise<void> {
		const match = /^\/mcp\/([^/?]+)\/([^/?]+)/.exec(request.url ?? "");
		if (!match?.[1] || !match[2]) return void response.writeHead(404).end();
		const agentId = decodePathSegment(match[1]);
		if (agentId === null) return void response.writeHead(400).end();
		const nonce = match[2];
		if (!secretsEqual(request.headers.authorization ?? "", `Bearer ${this.tokenFor(agentId, nonce)}`)) {
			return void response.writeHead(401).end();
		}
		if (this.nonces.get(nonce) !== agentId) return void response.writeHead(403).end();
		const sessionHeader = request.headers["mcp-session-id"];
		const sessionKey = typeof sessionHeader === "string" ? sessionHeader : `nonce:${nonce}`;
		const known = this.httpSessions.get(sessionKey);
		// The spec's answer to an unknown or ended session is 404, after which the client initializes again.
		if (typeof sessionHeader === "string" && known?.nonce !== nonce) return void response.writeHead(404).end();
		if (request.method === "DELETE") {
			if (known) {
				this.httpSessions.delete(sessionKey);
				abortAll(known.inflight);
			}
			return void response.writeHead(200).end();
		}
		// No server-initiated stream: the spec allows 405 for GET.
		if (request.method !== "POST") return void response.writeHead(405, { Allow: "POST, DELETE" }).end();
		let raw: Buffer | null;
		try {
			raw = await readBody(request, this.maxRequestBytes);
		} catch {
			// The client went away mid-upload; there is nobody to answer.
			return void response.destroy();
		}
		if (raw === null) {
			// WARN: the rest of the body is never read; closing the connection is what stops the upload.
			response.writeHead(413, { Connection: "close" }).end(() => request.destroy());
			return;
		}
		// WHY: the session may have been revoked while the body was uploading.
		if (this.nonces.get(nonce) !== agentId) return void response.writeHead(403).end();
		let body: unknown;
		try {
			body = JSON.parse(raw.toString("utf8"));
		} catch {
			response.writeHead(400, { "Content-Type": "application/json" }).end(JSON.stringify(rpcError(null, -32700, "Parse error")));
			return;
		}
		const messages = Array.isArray(body) ? body : [body];
		const initializing = messages.some((message) => isRecord(message) && message.method === "initialize");
		let session = this.httpSessions.get(sessionKey);
		// WHY: a DELETE may have ended the MCP session while the body was uploading.
		if (typeof sessionHeader === "string" && !session) return void response.writeHead(404).end();
		const headers: Record<string, string> = { "Content-Type": "application/json" };
		if (initializing && typeof sessionHeader !== "string") {
			const id = randomBytes(16).toString("hex");
			session = { nonce, inflight: new Map() };
			this.httpSessions.set(id, session);
			headers["Mcp-Session-Id"] = id;
		} else if (!session) {
			session = { nonce, inflight: new Map() };
			this.httpSessions.set(sessionKey, session);
		}
		const { inflight } = session;
		const owned = new Set<AbortController>();
		response.on("close", () => {
			if (response.writableFinished) return;
			for (const controller of owned) controller.abort();
		});
		const replies = (await Promise.all(messages.map((message) => this.handle(message, agentId, nonce, inflight, owned)))).filter(
			(reply): reply is object => reply !== null,
		);
		if (replies.length === 0) return void response.writeHead(202, headers).end();
		response.writeHead(200, headers).end(JSON.stringify(Array.isArray(body) ? replies : replies[0]));
	}

	private startSocket(): Promise<void> {
		if (IS_WINDOWS) {
			this.socketPath = `\\\\.\\pipe\\pi-acp-mcp-${process.pid}-${randomBytes(6).toString("hex")}`;
		} else {
			// mkdtemp creates the directory with mode 0700, so only this user can connect.
			this.socketDir = mkdtempSync(join(tmpdir(), "pi-acp-mcp-"));
			this.socketPath = join(this.socketDir, "mcp.sock");
		}
		const server = createNetServer((socket) => this.serveSocket(socket));
		this.socketServer = server;
		return new Promise((resolve, reject) => {
			server.once("error", reject);
			server.once("close", () => reject(new Error("Pi's MCP server was closed while starting")));
			server.listen(this.socketPath, () => resolve());
		});
	}

	private serveSocket(socket: Socket): void {
		let owner: { agentId: string; nonce: string } | null = null;
		const inflight: Inflight = new Map();
		socket.on("error", (error) => this.options.log?.(`mcp socket error: ${error.message}`));
		socket.on("close", () => {
			this.sockets.delete(socket);
			for (const controller of inflight.values()) controller.abort();
		});
		// WHY: the limit is read per line, so a request sent in the same chunk as the handshake gets the post-handshake limit.
		readLines(socket, () => (owner === null ? MAX_HELLO_BYTES : this.maxRequestBytes), (line) => {
			let message: unknown;
			try {
				message = JSON.parse(line);
			} catch {
				return;
			}
			if (owner === null) {
				// The relay's first line authenticates and names the agent and its session.
				const hello = isRecord(message) ? message : {};
				const { agent, session } = hello;
				if (
					typeof hello.token !== "string" ||
					typeof agent !== "string" ||
					typeof session !== "string" ||
					!secretsEqual(hello.token, this.tokenFor(agent, session)) ||
					this.nonces.get(session) !== agent
				) {
					socket.destroy();
					return;
				}
				owner = { agentId: agent, nonce: session };
				this.sockets.set(socket, owner);
				return;
			}
			if (!this.sockets.has(socket)) return;
			void this.handle(message, owner.agentId, owner.nonce, inflight).then((reply) => {
				if (reply && socket.writable) socket.write(`${JSON.stringify(reply)}\n`);
			});
		});
	}
}
