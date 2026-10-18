/**
 * ACP (Agent Client Protocol) v1 wire types and boundary parsers.
 *
 * Everything that crosses the stdio pipe is `unknown`; these parsers narrow it
 * exactly once, at the boundary, into the small domain types the provider
 * uses. Unknown fields and update kinds are ignored for forward compatibility.
 * Reference: https://agentclientprotocol.com/protocol/schema
 */

export const ACP_PROTOCOL_VERSION = 1;

/** ACP `auth_required` error code, returned by `session/new` or `session/prompt`. */
export const ACP_AUTH_REQUIRED = -32000;

export function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

const optionalString = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);

// --- JSON-RPC envelopes ------------------------------------------------------

/** JSON-RPC ids sent by the agent may be numbers or strings; ours are numbers. */
export type RpcId = number | string;

export interface RpcError {
	code: number;
	message: string;
}

/** Incoming JSON-RPC message, classified by shape. */
export type RpcIncoming =
	| { kind: "response"; id: RpcId; outcome: { ok: true; result: unknown } | { ok: false; error: RpcError } }
	| { kind: "request"; id: RpcId; method: string; params: unknown }
	| { kind: "notification"; method: string; params: unknown };

const isRpcId = (value: unknown): value is RpcId => typeof value === "number" || typeof value === "string";

export function parseRpcMessage(value: unknown): RpcIncoming | null {
	if (!isRecord(value)) return null;
	const { id, method } = value;
	if (typeof method === "string") {
		return isRpcId(id)
			? { kind: "request", id, method, params: value.params }
			: { kind: "notification", method, params: value.params };
	}
	if (!isRpcId(id)) return null;
	const error = value.error;
	if (isRecord(error)) {
		return {
			kind: "response",
			id,
			outcome: {
				ok: false,
				error: {
					code: typeof error.code === "number" ? error.code : -32603,
					message: optionalString(error.message) ?? "Unknown agent error",
				},
			},
		};
	}
	return "result" in value ? { kind: "response", id, outcome: { ok: true, result: value.result } } : null;
}

// --- session/update ----------------------------------------------------------

export type ToolCallStatus = "pending" | "in_progress" | "completed" | "failed";

/** The `session/update` variants the provider renders; everything else is dropped. */
export type AcpSessionUpdate =
	| { type: "agent_message_chunk"; text: string }
	| { type: "agent_thought_chunk"; text: string }
	| { type: "tool_call"; toolCallId: string; title: string; status: ToolCallStatus | undefined }
	| {
			type: "tool_call_update";
			toolCallId: string;
			title: string | undefined;
			status: ToolCallStatus | undefined;
			summary: string;
	  };

export interface AcpSessionNotification {
	sessionId: string;
	update: AcpSessionUpdate;
}

function parseToolCallStatus(value: unknown): ToolCallStatus | undefined {
	switch (value) {
		case "pending":
		case "in_progress":
		case "completed":
		case "failed":
			return value;
		default:
			return undefined;
	}
}

/** Text of a `ContentBlock`; non-text blocks (images, resources) render as nothing. */
function contentBlockText(block: unknown): string {
	return isRecord(block) && block.type === "text" ? (optionalString(block.text) ?? "") : "";
}

/** One-line summary of `ToolCallContent[]`: text output or the edited file paths. */
function toolContentSummary(content: unknown): string {
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const item of content) {
		if (!isRecord(item)) continue;
		if (item.type === "content") parts.push(contentBlockText(item.content));
		else if (item.type === "diff" && typeof item.path === "string") parts.push(`edited ${item.path}`);
	}
	return parts.filter(Boolean).join("; ");
}

/** Parses spec-shaped `{ sessionId, update: { sessionUpdate, ... } }` params. */
export function parseSessionNotification(params: unknown): AcpSessionNotification | null {
	if (!isRecord(params) || typeof params.sessionId !== "string" || !isRecord(params.update)) return null;
	const { sessionId, update: raw } = params;
	const update = ((): AcpSessionUpdate | null => {
		switch (raw.sessionUpdate) {
			case "agent_message_chunk":
			case "agent_thought_chunk":
				return { type: raw.sessionUpdate, text: contentBlockText(raw.content) };
			case "tool_call":
				if (typeof raw.toolCallId !== "string") return null;
				return {
					type: "tool_call",
					toolCallId: raw.toolCallId,
					title: optionalString(raw.title) ?? "tool",
					status: parseToolCallStatus(raw.status),
				};
			case "tool_call_update":
				if (typeof raw.toolCallId !== "string") return null;
				return {
					type: "tool_call_update",
					toolCallId: raw.toolCallId,
					title: optionalString(raw.title),
					status: parseToolCallStatus(raw.status),
					summary: toolContentSummary(raw.content),
				};
			default:
				return null;
		}
	})();
	return update ? { sessionId, update } : null;
}

// --- session/request_permission ---------------------------------------------

export type PermissionOptionKind = "allow_once" | "allow_always" | "reject_once" | "reject_always";

export interface AcpPermissionOption {
	optionId: string;
	name: string;
	kind: PermissionOptionKind;
}

export interface AcpPermissionRequest {
	/** Title of the tool call asking for permission, for display. */
	title: string;
	options: AcpPermissionOption[];
}

export type AcpPermissionOutcome = { outcome: "selected"; optionId: string } | { outcome: "cancelled" };

function parsePermissionKind(value: unknown): PermissionOptionKind | undefined {
	switch (value) {
		case "allow_once":
		case "allow_always":
		case "reject_once":
		case "reject_always":
			return value;
		default:
			return undefined;
	}
}

export function parsePermissionRequest(params: unknown): AcpPermissionRequest | null {
	if (!isRecord(params) || !Array.isArray(params.options)) return null;
	const options: AcpPermissionOption[] = [];
	for (const option of params.options) {
		if (!isRecord(option) || typeof option.optionId !== "string") continue;
		const kind = parsePermissionKind(option.kind);
		if (!kind) continue;
		options.push({ optionId: option.optionId, name: optionalString(option.name) ?? option.optionId, kind });
	}
	const title = (isRecord(params.toolCall) && optionalString(params.toolCall.title)) || "tool";
	return { title, options };
}

/** Selects the first offered option of the given kinds, in preference order. */
export function selectOption(request: AcpPermissionRequest, kinds: PermissionOptionKind[]): AcpPermissionOutcome {
	for (const kind of kinds) {
		const option = request.options.find((candidate) => candidate.kind === kind);
		if (option) return { outcome: "selected", optionId: option.optionId };
	}
	return { outcome: "cancelled" };
}

/** Allows once (never "always", so the agent's own permission memory stays untouched). */
export const autoApprove = (request: AcpPermissionRequest): AcpPermissionOutcome =>
	selectOption(request, ["allow_once", "allow_always"]);

/** Rejects once; used when no UI can ask the user. */
export const autoReject = (request: AcpPermissionRequest): AcpPermissionOutcome =>
	selectOption(request, ["reject_once", "reject_always"]);

// --- Responses ----------------------------------------------------------------

export interface AcpAuthMethod {
	id: string;
	name: string;
	description: string | undefined;
}

export interface AcpAgentInfo {
	name: string;
	version: string;
	/** Methods the client may pass to `authenticate` when the agent reports `auth_required`. */
	authMethods: AcpAuthMethod[];
}

function parseAuthMethods(value: unknown): AcpAuthMethod[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((method): AcpAuthMethod[] =>
		isRecord(method) && typeof method.id === "string"
			? [{ id: method.id, name: optionalString(method.name) ?? method.id, description: optionalString(method.description) }]
			: [],
	);
}

export function parseInitializeResult(result: unknown): AcpAgentInfo {
	const record = isRecord(result) ? result : {};
	const info = isRecord(record.agentInfo) ? record.agentInfo : {};
	return {
		name: optionalString(info.name) ?? "unknown-agent",
		version: optionalString(info.version) ?? "0.0.0",
		authMethods: parseAuthMethods(record.authMethods),
	};
}

export function parseNewSessionResult(result: unknown): string {
	if (!isRecord(result) || typeof result.sessionId !== "string" || result.sessionId.length === 0) {
		throw new Error("session/new did not return a sessionId");
	}
	return result.sessionId;
}

export type AcpStopReason = "end_turn" | "max_tokens" | "max_turn_requests" | "refusal" | "cancelled";

export function parseStopReason(result: unknown): AcpStopReason {
	const reason = isRecord(result) ? result.stopReason : undefined;
	switch (reason) {
		case "max_tokens":
		case "max_turn_requests":
		case "refusal":
		case "cancelled":
			return reason;
		default:
			return "end_turn";
	}
}
