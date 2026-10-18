/**
 * Pi extension entry point: registers the `acp` provider.
 *
 * Every agent configured under `acp.agents` in `~/.pi/agent/settings.json`
 * becomes a model, selected with `/model acp/<agent id>`. Each agent process is
 * spawned lazily on its first prompt, kept alive across prompts, and killed on
 * session shutdown so no orphans remain.
 */

import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { autoReject } from "./acp.ts";
import { loadSettings, resolveConfig } from "./config.ts";
import { createAcpProvider, type PermissionPolicy } from "./provider.ts";

type PermissionUi = Pick<ExtensionUIContext, "select">;

/** Asks the user via a select dialog; denies when no dialog-capable UI exists. */
function interactivePolicy(getUi: () => PermissionUi | undefined): PermissionPolicy {
	return async (request, { agent, signal }) => {
		const ui = getUi();
		if (!ui) return autoReject(request);
		const chosen = await ui.select(
			`${agent.name} permission: ${request.title}`,
			request.options.map((option) => option.name),
			signal ? { signal } : undefined,
		);
		const option = request.options.find((candidate) => candidate.name === chosen);
		return option ? { outcome: "selected", optionId: option.optionId } : { outcome: "cancelled" };
	};
}

export default function acpExtension(pi: ExtensionAPI): void {
	const debug = process.env.ACP_DEBUG;
	const log =
		debug === "1" || debug === "true" ? (message: string) => process.stderr.write(`[acp] ${message}\n`) : undefined;
	const settingsPath = join(homedir(), ".pi", "agent", "settings.json");
	const config = resolveConfig({ settings: loadSettings(settingsPath), env: process.env });

	let ui: PermissionUi | undefined;
	const handle = createAcpProvider({
		config,
		log,
		...(config.autoApproveTools ? {} : { permissionPolicy: interactivePolicy(() => ui) }),
	});
	pi.registerProvider(handle.provider);

	pi.on("session_start", (event, ctx) => {
		// A new Pi session (`/new`, `/resume`, `/fork`) gets new ACP sessions.
		handle.resetSession(ctx.cwd);
		ui = ctx.hasUI ? ctx.ui : undefined;
		if (event.reason === "startup" && config.agents.length === 0 && ctx.hasUI) {
			ctx.ui.notify(`ACP provider: no agents configured. Add "acp.agents" to ${settingsPath}.`, "warning");
		}
	});
	pi.on("session_shutdown", () => {
		ui = undefined;
		handle.dispose();
	});
}
