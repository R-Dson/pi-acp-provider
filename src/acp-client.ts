/**
 * JSON-RPC 2.0 client for an ACP agent over stdio (NDJSON framing).
 *
 * Owns the child process and the ACP session: lazy spawn, memoized
 * `initialize` handshake and `session/new`, one `session/prompt` at a time,
 * `session/cancel` on abort, and cleanup when the process dies. A crashed
 * process is replaced on next use with a fresh session. Server-initiated
 * requests (tool permissions) are answered by the injected handler.
 *
 * Nothing here is agent-specific: any command that speaks ACP v1 works.
 */

import { spawn, type ChildProcess, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createRequire } from "node:module";
import { createInterface } from "node:readline";
import {
	ACP_AUTH_REQUIRED,
	ACP_PROTOCOL_VERSION,
	RPC_INTERNAL_ERROR,
	isRecord,
	parseAvailableCommands,
	parseConfigOptionUpdate,
	parseInitializeResult,
	parseNewSessionResult,
	parseRpcMessage,
	parseSessionConfig,
	parseSessionUpdate,
	parseStopReason,
	type AcpAgentInfo,
	type AcpCommand,
	type AcpSessionConfig,
	type AcpSessionUpdate,
	type AcpStopReason,
	type RpcId,
} from "./acp.ts";
import type { AcpAgentConfig } from "./config.ts";

export type AcpConnectionState = "uninitialized" | "starting" | "ready" | "disposed";

export interface ServerRequest {
	method: string;
	params: unknown;
	/** The active prompt's abort signal; a cancelled turn must answer `cancelled`. */
	signal: AbortSignal | undefined;
}

/** Answers server-initiated requests such as `session/request_permission`. */
export type ServerRequestHandler = (request: ServerRequest) => Promise<unknown>;

export interface AcpClientOptions {
	agent: Pick<AcpAgentConfig, "name" | "command" | "args" | "env" | "authMethod">;
	cwd: string;
	/** Bounds `initialize`, `session/new`, and session setting changes. */
	handshakeTimeoutMs: number;
	onServerRequest: ServerRequestHandler;
	log?: ((message: string) => void) | undefined;
}

export interface PromptRequest {
	sessionId: string;
	/** `session/prompt` params (possibly rewritten by an `onPayload` hook). */
	payload: unknown;
	onUpdate: (update: AcpSessionUpdate) => void;
	/** Observes raw `session/update` params of this session before parsing. */
	onRawUpdate?: ((params: unknown) => void) | undefined;
	signal?: AbortSignal | undefined;
}

export type PromptResult = { kind: "finished"; stopReason: AcpStopReason } | { kind: "aborted" };

/** A JSON-RPC error, either received from the agent or sent back to it. */
export class AcpRpcError extends Error {
	readonly code: number;

	constructor(message: string, code: number) {
		super(message);
		this.name = "AcpRpcError";
		this.code = code;
	}
}

/** How long the next prompt waits for a cancelled turn's final response. */
const CANCEL_GRACE_MS = 5_000;
/** How long a SIGTERM'd agent gets before SIGKILL. */
const KILL_GRACE_MS = 3_000;
const PARENT_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;
const IS_WINDOWS = process.platform === "win32";
/** Sent as `clientInfo`; read from package.json so it tracks releases. */
const { name: CLIENT_NAME, version: CLIENT_VERSION } = createRequire(import.meta.url)("../package.json") as {
	name: string;
	version: string;
};

interface SessionState {
	config: AcpSessionConfig;
	/** The model the session started with, before any switch. */
	initialModel: string | undefined;
}

interface Connection {
	proc: ChildProcessWithoutNullStreams;
	state: "starting" | "ready";
	session: Promise<string> | null;
	/** Whether the current session has received a prompt (and so has history). */
	sessionUsed: boolean;
	/** Every session this process created, by session id. */
	sessions: Map<string, SessionState>;
	/** Slash commands the agent advertised, by session id. */
	commands: Map<string, AcpCommand[]>;
}

export interface EnsureSessionOptions {
	signal?: AbortSignal | undefined;
	/**
	 * Whether `auth_required` may trigger the configured `authenticate` method.
	 * Background work turns it off, since a method may wait on a browser login.
	 */
	authenticate?: boolean;
}

/** The session settings Pi controls. */
export type SessionSetting = keyof AcpSessionConfig;

interface PendingRequest {
	resolve: (result: unknown) => void;
	reject: (error: Error) => void;
}

const abortError = (): Error => new Error("Request was aborted");

/** Rejects as soon as `signal` aborts; the underlying work keeps running. */
function abortable<T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
	if (!signal) return promise;
	if (signal.aborted) return Promise.reject(abortError());
	return new Promise<T>((resolve, reject) => {
		const onAbort = () => reject(abortError());
		signal.addEventListener("abort", onAbort, { once: true });
		promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
	});
}

/** Resolves when `promise` settles or after `ms`, whichever comes first. */
function settleWithin(promise: Promise<unknown>, ms: number): Promise<void> {
	return new Promise((resolve) => {
		const timer = setTimeout(resolve, ms);
		timer.unref();
		const done = () => {
			clearTimeout(timer);
			resolve();
		};
		promise.then(done, done);
	});
}

/** Signals the agent's whole process group so its tool subprocesses die too. */
function signalTree(proc: ChildProcess, signal: NodeJS.Signals): void {
	if (proc.exitCode !== null || proc.signalCode !== null) return;
	try {
		if (proc.pid !== undefined && !IS_WINDOWS) process.kill(-proc.pid, signal);
		else proc.kill(signal);
	} catch {
		// Already gone.
	}
}

/** Agent processes of every client, killed if Pi exits or is signalled without a session shutdown. */
const running = new Set<ChildProcess>();

const killRunning = (): void => {
	for (const proc of running) signalTree(proc, "SIGTERM");
};

function onParentSignal(signal: NodeJS.Signals): void {
	killRunning();
	// Listening masks the default "terminate" disposition; re-raise it when
	// nobody else handles the signal.
	if (process.listenerCount(signal) === 1) {
		setParentHooks(false);
		process.kill(process.pid, signal);
	}
}

/** One set of hooks for all clients, so a lone listener can tell it is alone. */
function setParentHooks(on: boolean): void {
	const method = on ? "on" : "off";
	process[method]("exit", killRunning);
	for (const signal of PARENT_SIGNALS) process[method](signal, onParentSignal);
}

function track(proc: ChildProcess): void {
	if (running.size === 0) setParentHooks(true);
	running.add(proc);
}

function terminate(proc: ChildProcess): void {
	if (running.delete(proc) && running.size === 0) setParentHooks(false);
	signalTree(proc, "SIGTERM");
	setTimeout(() => signalTree(proc, "SIGKILL"), KILL_GRACE_MS).unref();
}

export class AcpClient {
	private readonly options: AcpClientOptions;
	private readonly log: (message: string) => void;
	private cwd: string;
	private conn: Connection | null = null;
	private ready: Promise<AcpAgentInfo> | null = null;
	private agentInfo: AcpAgentInfo | null = null;
	private disposed = false;
	private nextId = 1;
	private readonly pending = new Map<RpcId, PendingRequest>();
	private turn: PromptRequest | null = null;
	/** Settles once a cancelled turn's `session/prompt` response has arrived. */
	private draining: Promise<void> = Promise.resolve();

	constructor(options: AcpClientOptions) {
		this.options = options;
		this.cwd = options.cwd;
		this.log = options.log ?? (() => {});
	}

	get state(): AcpConnectionState {
		if (this.disposed) return "disposed";
		return this.conn?.state ?? "uninitialized";
	}

	/** What the agent reported in `initialize`, once the handshake completed. */
	get info(): AcpAgentInfo | null {
		return this.agentInfo;
	}

	/** Spawns the agent and completes the `initialize` handshake (idempotent). */
	ensureReady(signal?: AbortSignal): Promise<AcpAgentInfo> {
		if (this.disposed) return Promise.reject(new Error("AcpClient was disposed"));
		this.ready ??= this.handshake();
		return abortable(this.ready, signal);
	}

	/** Returns the current ACP session, creating one (and the process) on demand. */
	async ensureSession({ signal, authenticate = true }: EnsureSessionOptions = {}): Promise<string> {
		await this.ensureReady(signal);
		const conn = this.conn;
		if (!conn) throw new Error(`${this.options.agent.name} is not running`);
		conn.session ??= this.newSession(conn, authenticate).catch((error: unknown) => {
			conn.session = null;
			throw error;
		});
		return abortable(conn.session, signal);
	}

	/** Slash commands the current session advertised (`available_commands_update`). */
	async currentCommands(): Promise<AcpCommand[]> {
		const conn = this.conn;
		const sessionId = await conn?.session?.catch(() => undefined);
		return (sessionId && conn?.commands.get(sessionId)) || [];
	}

	/** The models and reasoning levels a live session offers. */
	sessionConfig(sessionId: string): AcpSessionConfig | undefined {
		return this.conn?.sessions.get(sessionId)?.config;
	}

	/** The model a live session started with, if the agent reported one. */
	initialModel(sessionId: string): string | undefined {
		return this.conn?.sessions.get(sessionId)?.initialModel;
	}

	/**
	 * Changes the session's model or reasoning level, via
	 * `session/set_config_option` or the legacy `session/set_model`.
	 */
	async setSessionSetting(sessionId: string, setting: SessionSetting, value: string, signal?: AbortSignal): Promise<void> {
		const session = this.conn?.sessions.get(sessionId);
		const select = session?.config[setting];
		if (!session || !select) throw new Error(`${this.options.agent.name} offers no ${setting} setting for this session`);
		if (select.currentValue === value) return;
		const timeout = this.options.handshakeTimeoutMs;
		if (select.configId === null) {
			await abortable(this.request("session/set_model", { sessionId, modelId: value }, timeout), signal);
			select.currentValue = value;
			return;
		}
		const result = await abortable(
			this.request("session/set_config_option", { sessionId, configId: select.configId, value }, timeout),
			signal,
		);
		session.config = parseSessionConfig(result, session.config);
	}

	/**
	 * Forgets the current session; the next `ensureSession` creates one in
	 * `cwd`. An unused session in the same directory is kept: it has no
	 * history, so it is as good as a new one.
	 */
	resetSession(cwd: string): void {
		const unchanged = cwd === this.cwd;
		this.cwd = cwd;
		if (!this.conn || (unchanged && !this.conn.sessionUsed)) return;
		this.conn.session = null;
		this.conn.sessionUsed = false;
	}

	/** Sends one `session/prompt` and resolves when the turn finishes or is aborted. */
	async prompt(request: PromptRequest): Promise<PromptResult> {
		if (this.turn) throw new Error(`Another ${this.options.agent.name} prompt is already in flight`);
		const { sessionId, signal } = request;
		this.turn = request;
		if (this.conn) this.conn.sessionUsed = true;
		try {
			await this.draining;
			if (signal?.aborted) return { kind: "aborted" };
			const response = this.request("session/prompt", request.payload);
			const finished = response.then((result): PromptResult => ({
				kind: "finished",
				stopReason: parseStopReason(result),
			}));
			if (!signal) return await finished;
			return await new Promise<PromptResult>((resolve, reject) => {
				const onAbort = () => {
					// Resolve now so the UI unblocks; the agent replies `cancelled`
					// later, and the next prompt waits for that (bounded).
					this.notify("session/cancel", { sessionId });
					this.draining = settleWithin(response, CANCEL_GRACE_MS);
					resolve({ kind: "aborted" });
				};
				signal.addEventListener("abort", onAbort, { once: true });
				finished.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
			});
		} finally {
			if (this.turn === request) this.turn = null;
		}
	}

	/** Kills the agent permanently; the client cannot be reused. */
	dispose(): void {
		this.disposed = true;
		if (this.conn) this.close(this.conn, new Error("ACP client was disposed"));
	}

	/** `session/new`, authenticating once with the configured method if the agent asks. */
	private async newSession(conn: Connection, authenticate: boolean): Promise<string> {
		const create = async () => {
			const { sessionId, config } = parseNewSessionResult(
				await this.request("session/new", { cwd: this.cwd, mcpServers: [] }, this.options.handshakeTimeoutMs),
			);
			conn.sessions.set(sessionId, { config, initialModel: config.model?.currentValue });
			return sessionId;
		};
		try {
			return await create();
		} catch (error) {
			const { authMethod } = this.options.agent;
			const authRequired = error instanceof AcpRpcError && error.code === ACP_AUTH_REQUIRED;
			if (!authRequired || !authMethod || !authenticate) throw error;
			this.log(`authenticating with method '${authMethod}'`);
			// No timeout: methods may wait on an interactive login (e.g. a browser).
			await this.request("authenticate", { methodId: authMethod });
			return create();
		}
	}

	private async handshake(): Promise<AcpAgentInfo> {
		const conn = this.connect();
		try {
			const result = await this.request(
				"initialize",
				{
					protocolVersion: ACP_PROTOCOL_VERSION,
					// Agents use their own file and terminal tools.
					clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
					clientInfo: { name: CLIENT_NAME, version: CLIENT_VERSION },
				},
				this.options.handshakeTimeoutMs,
			);
			const info = parseInitializeResult(result);
			// The agent answers with the version it will speak; we only speak v1.
			if (info.protocolVersion !== ACP_PROTOCOL_VERSION) {
				throw new Error(
					`${this.options.agent.name} speaks ACP v${info.protocolVersion}; ${CLIENT_NAME} supports v${ACP_PROTOCOL_VERSION}.`,
				);
			}
			conn.state = "ready";
			this.agentInfo = info;
			return info;
		} catch (error) {
			this.close(conn, error instanceof Error ? error : new Error(String(error)));
			throw error;
		}
	}

	private connect(): Connection {
		const { agent } = this.options;
		const proc = spawn(agent.command, agent.args, {
			cwd: this.cwd,
			env: { ...process.env, ...agent.env },
			// Own process group, so the whole tree can be killed at once.
			detached: !IS_WINDOWS,
		});
		const conn: Connection = { proc, state: "starting", session: null, sessionUsed: false, sessions: new Map(), commands: new Map() };
		this.conn = conn;
		track(proc);

		// EPIPE after a kill must not crash the host.
		proc.stdin.on("error", (error) => this.log(`stdin error: ${error.message}`));
		// Always drained, so a chatty agent never blocks on a full stderr pipe.
		proc.stderr.on("data", (chunk: Buffer) => this.log(`agent stderr: ${chunk.toString().trimEnd()}`));
		createInterface({ input: proc.stdout }).on("line", (line) => {
			if (this.conn !== conn) return;
			try {
				this.handleLine(line);
			} catch (error) {
				this.log(`line handler failed: ${String(error)}`);
			}
		});
		proc.on("error", (error: NodeJS.ErrnoException) => {
			this.close(
				conn,
				new Error(
					error.code === "ENOENT"
						? `${agent.name}: command '${agent.command}' not found. Install it or fix "command" for this agent in ~/.pi/agent/settings.json.`
						: `${agent.name}: failed to start '${agent.command}': ${error.message}`,
				),
			);
		});
		proc.on("exit", (code, signal) => {
			this.close(conn, new Error(`${agent.name} exited unexpectedly (code ${code}, signal ${signal})`));
		});
		return conn;
	}

	/** Tears down `conn` if it is still current; events from older processes are ignored. */
	private close(conn: Connection, reason: Error): void {
		if (this.conn !== conn) return;
		this.log(`closing connection: ${reason.message}`);
		this.conn = null;
		this.ready = null;
		this.draining = Promise.resolve();
		for (const entry of [...this.pending.values()]) entry.reject(reason);
		terminate(conn.proc);
	}

	private write(message: object): boolean {
		const stdin = this.conn?.proc.stdin;
		if (!stdin?.writable) return false;
		stdin.write(`${JSON.stringify(message)}\n`);
		return true;
	}

	private notify(method: string, params: unknown): void {
		this.write({ jsonrpc: "2.0", method, params });
	}

	private request(method: string, params: unknown, timeoutMs?: number): Promise<unknown> {
		const id = this.nextId++;
		return new Promise<unknown>((resolve, reject) => {
			let timer: NodeJS.Timeout | undefined;
			const settle = () => {
				this.pending.delete(id);
				clearTimeout(timer);
			};
			const entry: PendingRequest = {
				resolve: (result) => {
					settle();
					resolve(result);
				},
				reject: (error) => {
					settle();
					reject(error);
				},
			};
			this.pending.set(id, entry);
			if (timeoutMs !== undefined) {
				timer = setTimeout(() => entry.reject(new Error(`${method} timed out after ${timeoutMs}ms`)), timeoutMs);
			}
			if (!this.write({ jsonrpc: "2.0", id, method, params })) {
				entry.reject(new Error(`Cannot send ${method}: ${this.options.agent.name} is not running`));
			}
		});
	}

	private handleLine(line: string): void {
		if (!line.trim()) return;
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			this.log("discarded malformed NDJSON line");
			return;
		}
		const message = parseRpcMessage(parsed);
		if (!message) {
			this.log("discarded non-RPC line");
			return;
		}
		if (message.kind === "request") {
			void this.answerServerRequest(message.id, message.method, message.params);
		} else if (message.kind === "notification") {
			if (message.method === "session/update") this.dispatchUpdate(message.params);
		} else {
			const entry = this.pending.get(message.id);
			const { outcome } = message;
			if (!entry) this.log(`response for unknown request id ${message.id}`);
			else if (outcome.ok) entry.resolve(outcome.result);
			else entry.reject(new AcpRpcError(outcome.error.message, outcome.error.code));
		}
	}

	private dispatchUpdate(params: unknown): void {
		// Settings and commands can change outside a turn (e.g. the agent's own `/model`).
		if (!isRecord(params) || typeof params.sessionId !== "string") return;
		const { sessionId } = params;
		const conn = this.conn;
		if (conn) {
			const session = conn.sessions.get(sessionId);
			const config = session && parseConfigOptionUpdate(params, session.config);
			if (session && config) session.config = config;
			const commands = parseAvailableCommands(params);
			if (commands) conn.commands.set(sessionId, commands);
		}
		const turn = this.turn;
		if (!turn || sessionId !== turn.sessionId) return;
		turn.onRawUpdate?.(params);
		const update = parseSessionUpdate(params);
		if (update) turn.onUpdate(update);
	}

	private async answerServerRequest(id: RpcId, method: string, params: unknown): Promise<void> {
		try {
			const result = await this.options.onServerRequest({ method, params, signal: this.turn?.signal });
			this.write({ jsonrpc: "2.0", id, result: result ?? null });
		} catch (error) {
			this.write({
				jsonrpc: "2.0",
				id,
				error: {
					code: error instanceof AcpRpcError ? error.code : RPC_INTERNAL_ERROR,
					message: error instanceof Error ? error.message : String(error),
				},
			});
		}
	}
}
