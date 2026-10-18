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

import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import {
	ACP_AUTH_REQUIRED,
	ACP_PROTOCOL_VERSION,
	configOptionUpdate,
	parseAvailableCommands,
	parseInitializeResult,
	parseSessionConfig,
	parseNewSessionResult,
	parseRpcMessage,
	parseSessionNotification,
	parseStopReason,
	isRecord,
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
	/** Bounds `initialize` and `session/new`. */
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

export const RPC_METHOD_NOT_FOUND = -32601;
export const RPC_INVALID_PARAMS = -32602;
const RPC_INTERNAL_ERROR = -32603;

/** How long the next prompt waits for a cancelled turn's final response. */
const CANCEL_GRACE_MS = 5_000;
/** How long a SIGTERM'd agent gets before SIGKILL. */
const KILL_GRACE_MS = 3_000;
const PARENT_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;
const IS_WINDOWS = process.platform === "win32";

interface Connection {
	proc: ChildProcess;
	state: "starting" | "ready";
	session: Promise<string> | null;
	/** Whether the current session has received a prompt (and so has history). */
	sessionUsed: boolean;
	/** Settings of every session this process created, by session id. */
	configs: Map<string, AcpSessionConfig>;
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

interface ActiveTurn {
	sessionId: string;
	onUpdate: PromptRequest["onUpdate"];
	onRawUpdate: PromptRequest["onRawUpdate"];
	signal: AbortSignal | undefined;
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
		promise.then(
			() => {},
			() => {},
		).finally(() => {
			clearTimeout(timer);
			resolve();
		});
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

function terminate(proc: ChildProcess): void {
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
	private turn: ActiveTurn | null = null;
	/** Settles once a cancelled turn's `session/prompt` response has arrived. */
	private draining: Promise<void> = Promise.resolve();
	private parentHooks: [event: string, handler: () => void][] = [];

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
		return this.conn?.configs.get(sessionId);
	}

	/**
	 * Changes the session's model or reasoning level, via
	 * `session/set_config_option` or the legacy `session/set_model`.
	 */
	async setSessionSetting(sessionId: string, setting: SessionSetting, value: string, signal?: AbortSignal): Promise<void> {
		const conn = this.conn;
		const select = conn?.configs.get(sessionId)?.[setting];
		if (!conn || !select) throw new Error(`${this.options.agent.name} offers no ${setting} setting for this session`);
		if (select.currentValue === value) return;
		const timeout = this.options.handshakeTimeoutMs;
		if (select.configId === null) {
			await abortable(this.request("session/set_model", { sessionId, modelId: value }, timeout), signal);
			const config = conn.configs.get(sessionId);
			if (config?.model) config.model = { ...config.model, currentValue: value };
			return;
		}
		const result = await abortable(
			this.request("session/set_config_option", { sessionId, configId: select.configId, value }, timeout),
			signal,
		);
		const previous = conn.configs.get(sessionId);
		if (previous) conn.configs.set(sessionId, parseSessionConfig(result, previous));
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
		const turn: ActiveTurn = {
			sessionId,
			onUpdate: request.onUpdate,
			onRawUpdate: request.onRawUpdate,
			signal,
		};
		this.turn = turn;
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
			if (this.turn === turn) this.turn = null;
		}
	}

	/** Kills the agent permanently; the client cannot be reused. */
	dispose(): void {
		this.disposed = true;
		if (this.conn) this.close(this.conn, new Error("ACP client was disposed"));
		for (const [event, handler] of this.parentHooks) process.removeListener(event, handler);
		this.parentHooks = [];
	}

	/** `session/new`, authenticating once with the configured method if the agent asks. */
	private async newSession(conn: Connection, authenticate: boolean): Promise<string> {
		const create = async () => {
			const result = parseNewSessionResult(
				await this.request("session/new", { cwd: this.cwd, mcpServers: [] }, this.options.handshakeTimeoutMs),
			);
			conn.configs.set(result.sessionId, result.config);
			return result.sessionId;
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
					clientInfo: { name: "pi-acp-provider", version: "1.2.0" },
				},
				this.options.handshakeTimeoutMs,
			);
			const info = parseInitializeResult(result);
			// The agent answers with the version it will speak; we only speak v1.
			if (info.protocolVersion !== ACP_PROTOCOL_VERSION) {
				throw new Error(
					`${this.options.agent.name} speaks ACP v${info.protocolVersion}; pi-acp-provider supports v${ACP_PROTOCOL_VERSION}.`,
				);
			}
			conn.state = "ready";
			this.agentInfo = info;
			return this.agentInfo;
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
			stdio: ["pipe", "pipe", "pipe"],
			// Own process group, so the whole tree can be killed at once.
			detached: !IS_WINDOWS,
		});
		const conn: Connection = { proc, state: "starting", session: null, sessionUsed: false, configs: new Map(), commands: new Map() };
		this.conn = conn;
		this.installParentHooks();

		// EPIPE after a kill must not crash the host.
		proc.stdin?.on("error", (error) => this.log(`stdin error: ${error.message}`));
		proc.stderr?.on("data", (chunk: Buffer) => this.log(`agent stderr: ${chunk.toString().trimEnd()}`));
		if (proc.stdout) {
			createInterface({ input: proc.stdout }).on("line", (line) => {
				if (this.conn !== conn) return;
				try {
					this.handleLine(line);
				} catch (error) {
					this.log(`line handler failed: ${String(error)}`);
				}
			});
		}
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

	/** Kills the agent if Pi exits or is signalled without a session shutdown. */
	private installParentHooks(): void {
		if (this.parentHooks.length > 0) return;
		const kill = () => {
			if (this.conn) signalTree(this.conn.proc, "SIGTERM");
		};
		this.parentHooks.push(["exit", kill]);
		for (const signal of PARENT_SIGNALS) {
			const handler = () => {
				kill();
				// Listening masks the default "terminate" disposition; re-raise it
				// when nobody else handles the signal.
				if (process.listenerCount(signal) === 1) {
					this.dispose();
					process.kill(process.pid, signal);
				}
			};
			this.parentHooks.push([signal, handler]);
		}
		for (const [event, handler] of this.parentHooks) process.on(event, handler);
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
		switch (message.kind) {
			case "response": {
				const entry = this.pending.get(message.id);
				if (!entry) {
					this.log(`response for unknown request id ${message.id}`);
					return;
				}
				const { outcome } = message;
				if (outcome.ok) entry.resolve(outcome.result);
				else entry.reject(new AcpRpcError(outcome.error.message, outcome.error.code));
				return;
			}
			case "request":
				void this.answerServerRequest(message.id, message.method, message.params);
				return;
			case "notification":
				if (message.method === "session/update") this.dispatchUpdate(message.params);
				return;
			default: {
				const _exhaustive: never = message;
				void _exhaustive;
			}
		}
	}

	private dispatchUpdate(params: unknown): void {
		// Settings and commands can change outside a turn (e.g. the agent's own `/model`).
		const conn = this.conn;
		if (conn && isRecord(params) && typeof params.sessionId === "string") {
			const previous = conn.configs.get(params.sessionId);
			const configUpdate = configOptionUpdate(params);
			if (configUpdate && previous) conn.configs.set(params.sessionId, parseSessionConfig(configUpdate, previous));
			const commands = parseAvailableCommands(params);
			if (commands) conn.commands.set(params.sessionId, commands);
		}
		const turn = this.turn;
		if (!turn || !isRecord(params) || params.sessionId !== turn.sessionId) return;
		turn.onRawUpdate?.(params);
		const notification = parseSessionNotification(params);
		if (notification) turn.onUpdate(notification.update);
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
