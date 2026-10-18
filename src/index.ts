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
import { ACP_PROVIDER_ID, createAcpProvider, type PermissionPolicy } from "./provider.ts";

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

	// `/<agent id> <command>` runs a slash command the agent advertised
	// (e.g. Copilot's `/context`) by sending it to the agent as the prompt.
	for (const agent of config.agents) {
		pi.registerCommand(agent.id, {
			description: `Run a ${agent.name} slash command`,
			getArgumentCompletions: async (prefix) => {
				const items = (await handle.commands(agent.id))
					.filter((command) => command.name.startsWith(prefix.trim()))
					.map((command) => ({
						value: command.name,
						label: command.name,
						description: [command.description, command.hint && `(${command.hint})`].filter(Boolean).join(" "),
					}));
				return items.length > 0 ? items : null;
			},
			handler: async (args, ctx) => {
				if (!args.trim()) {
					const commands = await handle.commands(agent.id);
					const list = commands.map((command) => `/${agent.id} ${command.name}  ${command.description}`).join("\n");
					ctx.ui.notify(list || `${agent.name} has not advertised any commands yet.`, "info");
					return;
				}
				// Commands go to the agent, so make sure one of its models is active.
				const current = ctx.model;
				const onAgent = current?.provider === ACP_PROVIDER_ID && (current.id === agent.id || current.id.startsWith(`${agent.id}/`));
				if (!onAgent) {
					const model = handle.provider.getModels().find((candidate) => candidate.id === agent.id);
					if (!model || !(await pi.setModel(model))) {
						ctx.ui.notify(`Could not switch to ${agent.name}.`, "error");
						return;
					}
				}
				pi.sendUserMessage(`/${args.trim()}`);
			},
		});
	}

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
