/**
 * JSON-RPC 2.0 client for an ACP agent over stdio (NDJSON framing).
 *
 * Owns the child process and one ACP session per conversation: lazy spawn, memoized
 * `initialize` handshake and `session/new`, one `session/prompt` at a time per session,
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
	parsePromptUsage,
	parseRpcMessage,
	parseSessionConfig,
	parseSessionUpdate,
	parseStopReason,
	type AcpAgentInfo,
	type AcpCommand,
	type AcpSessionConfig,
	type AcpSessionUpdate,
	type AcpStopReason,
	type AcpUsage,
	type RpcId,
} from "./acp.ts";
import type { AcpAgentConfig } from "./config.ts";

export type AcpConnectionState = "uninitialized" | "starting" | "ready" | "disposed";

export interface ServerRequest {
	method: string;
	params: unknown;
	/** `params.sessionId`, when it names a live session of this client. */
	sessionId: string | undefined;
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
	/**
	 * Called once for every session that stops being usable: replaced, reset, or lost with its
	 * process. Its later traffic is refused, so anything bound to it (an MCP nonce) can go.
	 */
	onSessionEnd?: ((sessionId: string) => void) | undefined;
	/** How long a cancelled prompt gets to answer before its session is replaced (default 5 s). */
	cancelGraceMs?: number | undefined;
	log?: ((message: string) => void) | undefined;
}

export interface PromptRequest {
	sessionId: string;
	/** `session/prompt` params (possibly rewritten by an `onPayload` hook). */
	payload: unknown;
	onUpdate: (update: AcpSessionUpdate) => void;
	/** Observes raw `session/update` params of this session before parsing. */
	onRawUpdate?: ((params: unknown) => void) | undefined;
	/**
	 * Called once `session/prompt` is written, after any cancelled prompt of the session has
	 * drained: from here on, the session's traffic belongs to this prompt.
	 */
	onSent?: (() => void) | undefined;
	signal?: AbortSignal | undefined;
	/** The session instance (`AcpClient.instance`) the prompt is for; refused once that one ended. */
	instance?: SessionInstance | undefined;
}

declare const sessionInstanceBrand: unique symbol;
/**
 * One session of one agent process. Session ids repeat across processes (a respawned agent numbers its
 * sessions again), so an id alone can name another conversation's new session.
 */
export interface SessionInstance {
	readonly [sessionInstanceBrand]: true;
}

export type PromptResult = { kind: "finished"; stopReason: AcpStopReason; usage: AcpUsage | undefined } | { kind: "aborted" };

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
/** Drain timeouts in a row after which the agent process is restarted rather than given a new session. */
const MAX_DRAIN_TIMEOUTS = 2;
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

/** The current session of one conversation (`EnsureSessionOptions.conversation`). */
interface SessionSlot {
	session: Promise<string> | null;
	/** The id `session` resolved to. */
	id: string | undefined;
	/** `setup.key` the current session was created with; "" for a bare session. */
	key: string;
	/** Whether the current session has received a prompt (and so has history). */
	used: boolean;
	/** Whether the next session opened replaces one whose cancelled prompt never answered. */
	replacing: boolean;
	/** The `session` promise opened while `replacing`, until it settles. */
	replacement: Promise<string> | null;
}

interface Connection {
	proc: ChildProcessWithoutNullStreams;
	state: "starting" | "ready";
	/** Each conversation's current session, by conversation. */
	slots: Map<string, SessionSlot>;
	/** Every session this process created, by session id. */
	sessions: Map<string, SessionState>;
	/** Slash commands the agent advertised, by session id. */
	commands: Map<string, AcpCommand[]>;
	/** Sessions of this process that ended; their traffic is dropped or answered `cancelled`. */
	ended: Set<string>;
}

export interface EnsureSessionOptions {
	signal?: AbortSignal | undefined;
	/**
	 * Whether `auth_required` may trigger the configured `authenticate` method.
	 * Background work turns it off, since a method may wait on a browser login.
	 */
	authenticate?: boolean;
	/**
	 * Extra `session/new` params. A current session created with another key is
	 * replaced while it has no history (e.g. a bare discovery session), and kept
	 * once it has one.
	 */
	setup?: SessionSetup | undefined;
	/** Which conversation's session: each has its own, so their prompts can run at once (default ""). */
	conversation?: string | undefined;
}

export interface SessionSetup {
	key: string;
	params: (info: AcpAgentInfo) => Record<string, unknown>;
	/** Called with the `params` result once its `session/new` returned a session id, or failed (undefined). */
	settled?: ((params: Record<string, unknown>, sessionId: string | undefined) => void) | undefined;
}

/** The session settings Pi controls. */
export type SessionSetting = keyof AcpSessionConfig;

interface PendingRequest {
	resolve: (result: unknown) => void;
	reject: (error: Error) => void;
}

/** A cancelled prompt waiting for its `cancelled` response. */
interface Drain {
	/** The `session/prompt` request id. */
	requestId: RpcId;
	/** True once the prompt answered, false when the cancel grace ran out first. */
	settled: Promise<boolean>;
}

const abortError = (): Error => new Error("Request was aborted");
const errorText = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/** Rejects as soon as `signal` aborts; the underlying work keeps running. */
export function abortable<T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
	if (!signal) return promise;
	if (signal.aborted) {
		// WHY: the work may already have failed (a hook that aborted the request and rejected); unobserved,
		// its rejection would be unhandled.
		promise.catch(() => {});
		return Promise.reject(abortError());
	}
	return new Promise<T>((resolve, reject) => {
		const onAbort = () => reject(abortError());
		signal.addEventListener("abort", onAbort, { once: true });
		promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
	});
}

/** Resolves true when `promise` settles, or false after `ms`, whichever comes first. */
function settleWithin(promise: Promise<unknown>, ms: number): Promise<boolean> {
	return new Promise((resolve) => {
		const timer = setTimeout(() => resolve(false), ms);
		timer.unref();
		const done = () => {
			clearTimeout(timer);
			resolve(true);
		};
		promise.then(done, done);
	});
}

/** Signals the agent's whole process group so its tool subprocesses die too. */
function signalTree(proc: ChildProcess, signal: NodeJS.Signals): void {
	try {
		// WHY: the group is signalled even after the leader exited, since members that ignored SIGTERM
		// (a stdio MCP relay, a tool subprocess) outlive it; the pgid stays reserved while any member
		// lives, and ESRCH means none does.
		if (proc.pid !== undefined && !IS_WINDOWS) process.kill(-proc.pid, signal);
		else if (proc.exitCode === null && proc.signalCode === null) proc.kill(signal);
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
	/** Sessions with a `prompt` call in progress, including its wait for a drain. */
	private readonly busy = new Set<string>();
	/**
	 * The prompt each session's updates and permission requests belong to. Set only once the
	 * prompt is sent: ACP updates carry no prompt id, so this is what keeps a cancelled prompt's
	 * late traffic out of the next one.
	 */
	private readonly turns = new Map<string, PromptRequest>();
	/** Cancelled prompts whose `cancelled` response has not arrived yet, by session id. */
	private readonly drains = new Map<string, Drain>();
	/** The settings changes queued on each session, by session id: each waits for the ones before it. */
	private readonly settingChanges = new Map<string, Promise<void>>();
	/** Drain timeouts in a row; a drain that ends in time, or a restart, resets it. */
	private drainTimeouts = 0;
	/** The drain-timeout recovery in progress for each conversation; its concurrent `ensureSession` calls share it. */
	private readonly recoveries = new Map<string, Promise<void>>();

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

	/** Returns the conversation's current ACP session, creating one (and the process) on demand. */
	async ensureSession(options: EnsureSessionOptions = {}): Promise<string> {
		const { signal, authenticate = true, setup, conversation = "" } = options;
		let recovery = this.recoveries.get(conversation);
		if (!recovery) {
			const started: Promise<void> = this.recoverDrain(conversation).finally(() => {
				if (this.recoveries.get(conversation) === started) this.recoveries.delete(conversation);
			});
			this.recoveries.set(conversation, (recovery = started));
		}
		await abortable(recovery, signal);
		const info = await this.ensureReady(signal);
		const conn = this.conn;
		if (!conn) throw new Error(`${this.options.agent.name} is not running`);
		let slot = conn.slots.get(conversation);
		if (!slot) conn.slots.set(conversation, (slot = { session: null, id: undefined, key: "", used: false, replacing: false, replacement: null }));
		const key = setup?.key ?? "";
		if (slot.session && setup && slot.key !== key && !slot.used) this.dropSession(conn, slot);
		let session = slot.session;
		if (!session) {
			const owner = slot;
			owner.key = key;
			const params = setup?.params(info) ?? {};
			const created: Promise<string> = this.newSession(conn, authenticate, params).then(
				(sessionId) => {
					if (owner.replacement === created) owner.replacement = null;
					if (owner.session === created) owner.id = sessionId;
					setup?.settled?.(params, sessionId);
					return sessionId;
				},
				(error: unknown) => {
					setup?.settled?.(params, undefined);
					if (owner.session === created) owner.session = null;
					// WHY: decided here, on the shared operation, so the restart happens even when every
					// caller waiting on this session has aborted. An agent that cannot open the session
					// replacing a stuck one is not trusted to recover by itself.
					if (owner.replacement === created) {
						owner.replacement = null;
						if (this.conn === conn && !this.disposed) {
							this.log(`new session after a drain timeout failed (${errorText(error)}); restarting the agent`);
							this.close(conn, new Error("restarted after a drain timeout"));
						}
					}
					throw error;
				},
			);
			owner.session = session = created;
			if (owner.replacing) {
				owner.replacing = false;
				owner.replacement = created;
			}
		}
		const replacement = slot.replacement === session;
		try {
			return await abortable(session, signal);
		} catch (error) {
			// The failed replacement restarted the agent; the session is retried once on the new process.
			if (!replacement || signal?.aborted || this.disposed) throw error;
			return this.ensureSession(options);
		}
	}

	/**
	 * Waits for the conversation's current session's cancelled prompt, if any. When it does not answer
	 * within the cancel grace, rejects it and ends the session so the next one starts clean; on the
	 * second timeout in a row restarts the process instead.
	 */
	private async recoverDrain(conversation: string): Promise<void> {
		const conn = this.conn;
		const slot = conn?.slots.get(conversation);
		const sessionId = await slot?.session?.catch(() => undefined);
		const drain = sessionId === undefined ? undefined : this.drains.get(sessionId);
		if (!conn || !slot || sessionId === undefined || !drain) return;
		// A drain that ends in time resets the timeout streak where it settles (in `prompt`).
		if (await drain.settled) return;
		if (this.conn !== conn || this.drains.get(sessionId) !== drain) return;
		this.drains.delete(sessionId);
		this.pending.get(drain.requestId)?.reject(new Error("The cancelled prompt did not end in time"));
		this.drainTimeouts++;
		if (this.drainTimeouts >= MAX_DRAIN_TIMEOUTS) {
			this.log(`session ${sessionId} ignored session/cancel again; restarting the agent`);
			this.close(conn, new Error("restarted after repeated drain timeouts"));
			return;
		}
		this.log(`session ${sessionId} ignored session/cancel; starting a new session`);
		this.dropSession(conn, slot);
		slot.replacing = true;
	}

	/** Forgets the slot's current session and ends it once it exists, so its traffic is refused. */
	private dropSession(conn: Connection, slot: SessionSlot): void {
		const session = slot.session;
		slot.session = null;
		slot.id = undefined;
		slot.used = false;
		void session?.then(
			(sessionId) => this.endSession(conn, sessionId),
			() => {},
		);
	}

	private endSession(conn: Connection, sessionId: string): void {
		if (conn.ended.has(sessionId)) return;
		conn.ended.add(sessionId);
		conn.sessions.delete(sessionId);
		conn.commands.delete(sessionId);
		const drain = this.drains.get(sessionId);
		if (drain) {
			this.drains.delete(sessionId);
			this.pending.get(drain.requestId)?.reject(new Error(`Session ${sessionId} ended before its cancelled prompt did`));
		}
		this.turns.delete(sessionId);
		try {
			this.options.onSessionEnd?.(sessionId);
		} catch (error) {
			this.log(`onSessionEnd failed: ${errorText(error)}`);
		}
	}

	/** Slash commands the newest conversation's current session advertised (`available_commands_update`). */
	async currentCommands(): Promise<AcpCommand[]> {
		const conn = this.conn;
		for (const slot of [...(conn?.slots.values() ?? [])].reverse()) {
			const sessionId = await slot.session?.catch(() => undefined);
			const commands = sessionId === undefined ? undefined : conn?.commands.get(sessionId);
			if (commands) return commands;
		}
		return [];
	}

	/** The current instance of the live session `sessionId`, to bind later calls to it. */
	instance(sessionId: string): SessionInstance | undefined {
		return this.conn?.sessions.get(sessionId) as SessionInstance | undefined;
	}

	/** Throws when `instance` is given and the live session `sessionId` is no longer that instance. */
	private assertInstance(sessionId: string, instance: SessionInstance | undefined): void {
		if (instance !== undefined && this.instance(sessionId) !== instance) {
			throw new Error(`${this.options.agent.name} session ${sessionId} ended (the agent restarted or the session was replaced)`);
		}
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
	 * `session/set_config_option` or the legacy `session/set_model`, after every change queued on the
	 * session before it; an aborted caller stops waiting, but a change already sent still lands.
	 * WARN: the cached value is compared only once the earlier changes landed: a caller that left mid-change
	 * must not make a later caller skip a write the agent still needs.
	 */
	async setSessionSetting(
		sessionId: string,
		setting: SessionSetting,
		value: string,
		signal?: AbortSignal,
		instance?: SessionInstance,
	): Promise<void> {
		signal?.throwIfAborted();
		this.assertInstance(sessionId, instance);
		const change = (this.settingChanges.get(sessionId) ?? Promise.resolve()).then(() => this.applySetting(sessionId, setting, value, instance));
		const settled = change.catch(() => {});
		this.settingChanges.set(sessionId, settled);
		void settled.then(() => {
			if (this.settingChanges.get(sessionId) === settled) this.settingChanges.delete(sessionId);
		});
		await abortable(change, signal);
	}

	private async applySetting(sessionId: string, setting: SessionSetting, value: string, instance: SessionInstance | undefined): Promise<void> {
		this.assertInstance(sessionId, instance);
		const session = this.conn?.sessions.get(sessionId);
		const select = session?.config[setting];
		if (!session || !select) throw new Error(`${this.options.agent.name} offers no ${setting} setting for this session`);
		if (select.currentValue === value) return;
		const timeout = this.options.handshakeTimeoutMs;
		if (select.configId === null) {
			await this.request("session/set_model", { sessionId, modelId: value }, timeout);
			select.currentValue = value;
			return;
		}
		const result = await this.request("session/set_config_option", { sessionId, configId: select.configId, value }, timeout);
		session.config = parseSessionConfig(result, session.config);
	}

	/**
	 * Forgets the current session of `conversation`, or of every conversation; the next `ensureSession`
	 * creates one in `cwd`. An unused session in the same directory is kept: it has no history, so it is
	 * as good as a new one.
	 */
	resetSession(cwd: string, conversation?: string): void {
		const unchanged = cwd === this.cwd;
		this.cwd = cwd;
		const conn = this.conn;
		if (!conn) return;
		for (const [name, slot] of conn.slots) {
			if (conversation !== undefined && name !== conversation) continue;
			if (!unchanged || slot.used) this.dropSession(conn, slot);
		}
	}

	/** Sends one `session/prompt` and resolves when the turn finishes or is aborted. */
	async prompt(request: PromptRequest): Promise<PromptResult> {
		const { sessionId, signal } = request;
		this.assertInstance(sessionId, request.instance);
		if (this.busy.has(sessionId)) throw new Error(`Another ${this.options.agent.name} prompt is already in flight`);
		this.busy.add(sessionId);
		for (const slot of this.conn?.slots.values() ?? []) if (slot.id === sessionId) slot.used = true;
		try {
			if (!(await this.drained(sessionId))) {
				throw new Error(`${this.options.agent.name} session ${sessionId} is still running a cancelled prompt; open a new session`);
			}
			if (signal?.aborted) return { kind: "aborted" };
			if (this.conn?.ended.has(sessionId)) throw new Error(`${this.options.agent.name} session ${sessionId} has ended`);
			this.assertInstance(sessionId, request.instance);
			this.turns.set(sessionId, request);
			const { id: requestId, response } = this.send("session/prompt", request.payload);
			request.onSent?.();
			const finished = response.then((result): PromptResult => ({
				kind: "finished",
				stopReason: parseStopReason(result),
				usage: parsePromptUsage(result),
			}));
			if (!signal) return await finished;
			return await new Promise<PromptResult>((resolve, reject) => {
				const onAbort = () => {
					// Resolve now so the UI unblocks; the agent replies `cancelled`
					// later, and the next prompt waits for that (bounded).
					this.notify("session/cancel", { sessionId });
					if (this.turns.get(sessionId) === request) this.turns.delete(sessionId);
					const drain: Drain = { requestId, settled: settleWithin(response, this.options.cancelGraceMs ?? CANCEL_GRACE_MS) };
					this.drains.set(sessionId, drain);
					// WARN: a timed-out drain stays in `drains` (its session's traffic is still refused) until
					// `recoverDrain` replaces the session.
					void drain.settled.then((inTime) => {
						if (!inTime || this.drains.get(sessionId) !== drain) return;
						this.drains.delete(sessionId);
						this.drainTimeouts = 0;
					});
					resolve({ kind: "aborted" });
				};
				signal.addEventListener("abort", onAbort, { once: true });
				finished.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
			});
		} finally {
			this.busy.delete(sessionId);
			if (this.turns.get(sessionId) === request) this.turns.delete(sessionId);
		}
	}

	/** Waits for the session's cancelled prompt, if any, to answer; false when the cancel grace ran out. */
	private async drained(sessionId: string): Promise<boolean> {
		return (await this.drains.get(sessionId)?.settled) ?? true;
	}

	/** Kills the agent permanently; the client cannot be reused. */
	dispose(): void {
		this.disposed = true;
		if (this.conn) this.close(this.conn, new Error("ACP client was disposed"));
	}

	/** `session/new`, authenticating once with the configured method if the agent asks. */
	private async newSession(conn: Connection, authenticate: boolean, extra: Record<string, unknown>): Promise<string> {
		const create = async () => {
			const { sessionId, config } = parseNewSessionResult(
				await this.request("session/new", { cwd: this.cwd, mcpServers: [], ...extra }, this.options.handshakeTimeoutMs),
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
		const conn: Connection = {
			proc,
			state: "starting",
			slots: new Map(),
			sessions: new Map(),
			commands: new Map(),
			ended: new Set(),
		};
		this.conn = conn;
		track(proc);

		// EPIPE after a kill must not crash the host.
		proc.stdin.on("error", (error) => this.log(`stdin error: ${error.message}`));
		// Always drained, so a chatty agent never blocks on a full stderr pipe.
		proc.stderr.on("data", (chunk: Buffer) => this.log(`agent stderr: ${chunk.toString().trimEnd()}`));
		createInterface({ input: proc.stdout }).on("line", (line) => {
			if (this.conn !== conn) return;
			try {
				this.handleLine(conn, line);
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
		this.drainTimeouts = 0;
		for (const sessionId of [...conn.sessions.keys()]) this.endSession(conn, sessionId);
		this.drains.clear();
		for (const entry of [...this.pending.values()]) entry.reject(reason);
		terminate(conn.proc);
	}

	private write(message: object, conn: Connection | null = this.conn): boolean {
		const stdin = conn?.proc.stdin;
		if (!stdin?.writable) return false;
		stdin.write(`${JSON.stringify(message)}\n`);
		return true;
	}

	private notify(method: string, params: unknown): void {
		this.write({ jsonrpc: "2.0", method, params });
	}

	private request(method: string, params: unknown, timeoutMs?: number): Promise<unknown> {
		return this.send(method, params, timeoutMs).response;
	}

	private send(method: string, params: unknown, timeoutMs?: number): { id: RpcId; response: Promise<unknown> } {
		const id = this.nextId++;
		const response = new Promise<unknown>((resolve, reject) => {
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
		return { id, response };
	}

	private handleLine(conn: Connection, line: string): void {
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
			void this.answerServerRequest(conn, message.id, message.method, message.params);
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
		if (conn?.ended.has(sessionId)) return;
		if (conn) {
			const session = conn.sessions.get(sessionId);
			const config = session && parseConfigOptionUpdate(params, session.config);
			if (session && config) session.config = config;
			// WHY: a draining prompt's command list is late output of a cancelled turn, so it must not
			// replace the one the next prompt sees. Config updates are still taken: they report the
			// agent's actual settings, which the next prompt re-applies anyway.
			const commands = this.drains.has(sessionId) ? null : parseAvailableCommands(params);
			if (commands) conn.commands.set(sessionId, commands);
		}
		const turn = this.turns.get(sessionId);
		if (!turn) return;
		turn.onRawUpdate?.(params);
		const update = parseSessionUpdate(params);
		if (update) turn.onUpdate(update);
	}

	private async answerServerRequest(conn: Connection, id: RpcId, method: string, params: unknown): Promise<void> {
		const sessionId = isRecord(params) && typeof params.sessionId === "string" ? params.sessionId : undefined;
		const permission = method === "session/request_permission";
		// WHY: a cancelled prompt must answer its permission requests `cancelled`; while it drains,
		// the session has no current prompt whose signal could say so. An ended, unknown or missing
		// session has no prompt Pi could approve anything for.
		const usable = (): boolean =>
			sessionId !== undefined && this.conn === conn && conn.sessions.has(sessionId) && !this.drains.has(sessionId);
		const cancelled = { jsonrpc: "2.0", id, result: { outcome: { outcome: "cancelled" } } };
		// WARN: replies go to the connection the request came on, never `this.conn`: a restart while the
		// policy waits would otherwise hand this id to the new process, where it can match its own request.
		const reply = (message: object): void => {
			if (this.conn !== conn) return;
			this.write(permission && !usable() ? cancelled : message, conn);
		};
		if (permission && !usable()) {
			this.write(cancelled, conn);
			return;
		}
		const live = sessionId !== undefined && conn.sessions.has(sessionId);
		try {
			const turn = sessionId === undefined ? undefined : this.turns.get(sessionId);
			const result = await this.options.onServerRequest({
				method,
				params,
				sessionId: live ? sessionId : undefined,
				signal: turn?.signal,
			});
			reply({ jsonrpc: "2.0", id, result: result ?? null });
		} catch (error) {
			reply({
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
