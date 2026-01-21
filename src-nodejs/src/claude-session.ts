import { randomUUID } from 'crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import type {
  HookInput,
  PermissionResult,
  SDKAssistantMessage,
  SDKMessage,
  SDKPartialAssistantMessage,
  SDKResultMessage,
  SDKUserMessage,
} from '@anthropic-ai/claude-agent-sdk';
import type { JsonRpcHandler } from './rpc.js';
import type { SessionState, TurnStartParams, TurnInput } from './types.js';

// Map reasoning effort to maxThinkingTokens for the Claude SDK
// Note: Claude Opus 4.5 has a max of 64000 total output tokens (thinking + response)
// We leave headroom for response tokens when using extended thinking
const EFFORT_TO_THINKING_TOKENS: Record<string, number | undefined> = {
  default: undefined, // Let SDK decide
  low: 4096,
  medium: 16384,
  high: 60000, // Leave 4000 tokens for response (max_tokens = 64000)
};

type StoredThread = {
  id: string;
  cwd: string;
  createdAt: number;
  updatedAt: number;
  preview?: string;
  sessionId?: string | null;
  turns: StoredTurn[];
};

type StoredTurn = {
  id: string;
  createdAt: number;
  items: StoredItem[];
};

type FileChange = {
  path: string;
  kind?: 'add' | 'modify' | 'delete';
  diff?: string;
};

type StoredItem =
  | {
      id: string;
      type: 'userMessage';
      content: TurnInput[];
    }
  | {
      id: string;
      type: 'agentMessage';
      text: string;
    }
  | {
      id: string;
      type: 'reasoning';
      summary?: string;
      content?: string;
    }
  | {
      id: string;
      type: 'commandExecution';
      command?: string | string[];
      cwd?: string;
      status?: string;
      aggregatedOutput?: string;
    }
  | {
      id: string;
      type: 'fileChange';
      changes?: FileChange[];
      cwd?: string;
      status?: string;
      diff?: string;
    };

type ToolKind = 'commandExecution' | 'fileChange';
type ToolOutputMethod = 'item/commandExecution/outputDelta' | 'item/fileChange/outputDelta';

type ClaudeSessionConfig = {
  workspaceId: string;
  dataDir?: string | null;
  claudeCodePath?: string | null;
};

type SessionUsage = {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  totalCostUsd: number;
  turnCount: number;
  startedAt: number;
};

type TokenUsageBreakdown = {
  totalTokens: number;
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  reasoningOutputTokens: number;
};

type ThreadTokenUsage = {
  total: TokenUsageBreakdown;
  last: TokenUsageBreakdown;
  modelContextWindow: number | null;
};

type TurnContext = {
  threadId: string;
  turnId: string;
  reasoningItemId: string | null;
  reasoningBlockIndex: number | null;
  toolOutputMethods: Map<string, ToolOutputMethod>;
  abortController: AbortController;
  activeQuery: { interrupt?: () => Promise<void> } | null;
  model: string | null;
};

// Claude model context windows (in tokens)
// All current Claude models support 200K context window
// Sonnet 4.5 also has a 1M token context window in beta
const MODEL_CONTEXT_WINDOWS: Record<string, number> = {
  'claude-opus-4-5-20251101': 200000,
  'claude-sonnet-4-5-20250929': 200000,
  'claude-sonnet-4-20250514': 200000,
  'claude-3-7-sonnet-20250219': 200000,
  'claude-3-5-sonnet-20241022': 200000,
  'claude-3-5-sonnet-20240620': 200000,
  'claude-3-5-haiku-20241022': 200000,
  'claude-3-opus-20240229': 200000,
  'claude-3-sonnet-20240229': 200000,
  'claude-3-haiku-20240307': 200000,
};

function getModelContextWindow(model: string | undefined): number | null {
  if (!model) return 200000; // Default to 200K
  // Try exact match
  if (MODEL_CONTEXT_WINDOWS[model]) {
    return MODEL_CONTEXT_WINDOWS[model];
  }
  // Try prefix match (e.g., "claude-opus-4-5" matches "claude-opus-4-5-20251101")
  for (const [key, value] of Object.entries(MODEL_CONTEXT_WINDOWS)) {
    if (model.startsWith(key.split('-').slice(0, -1).join('-'))) {
      return value;
    }
  }
  // Default to 200K for unknown Claude models
  return 200000;
}

type ToolStartEvent = {
  method: 'item/started';
  params: {
    threadId: string;
    turnId: string;
    item: StoredItem;
  };
};

type ToolOutputEvent = {
  method: ToolOutputMethod;
  params: {
    threadId: string;
    turnId: string;
    itemId: string;
    delta: string;
  };
};

type TurnErrorEvent = {
  method: 'error';
  params: {
    threadId: string;
    turnId: string;
    error: { message: string };
    willRetry: boolean;
  };
};

export function mapDecisionToPermissionResult(
  response: unknown,
  toolUseId?: string,
  originalInput?: Record<string, unknown>,
): PermissionResult {
  const data = response as { decision?: string; approved?: boolean } | null;
  // Claude SDK expects { behavior: 'allow', updatedInput: Record } for allow
  if (data?.approved === true || data?.decision === 'accept') {
    return {
      behavior: 'allow',
      updatedInput: originalInput ?? {},
      toolUseID: toolUseId,
    };
  }
  if (data?.decision === 'decline') {
    return {
      behavior: 'deny',
      message: 'Tool use denied by user.',
      toolUseID: toolUseId,
    };
  }
  return {
    behavior: 'deny',
    message: 'Tool use denied.',
    toolUseID: toolUseId,
  };
}

export function buildToolStartEvent(
  threadId: string,
  turnId: string,
  item: StoredItem,
): ToolStartEvent {
  return {
    method: 'item/started',
    params: { threadId, turnId, item },
  };
}

export function buildToolOutputEvent(
  threadId: string,
  turnId: string,
  itemId: string,
  delta: string,
  method: ToolOutputMethod,
): ToolOutputEvent {
  return {
    method,
    params: { threadId, turnId, itemId, delta },
  };
}

export function buildTurnErrorEvent(
  threadId: string,
  turnId: string,
  message: string,
): TurnErrorEvent {
  return {
    method: 'error',
    params: {
      threadId,
      turnId,
      error: { message },
      willRetry: false,
    },
  };
}

function formatToolOutput(toolResponse: unknown): string {
  if (toolResponse === null || toolResponse === undefined) {
    return '';
  }
  if (typeof toolResponse === 'string') {
    return toolResponse;
  }
  if (typeof toolResponse === 'number' || typeof toolResponse === 'boolean') {
    return String(toolResponse);
  }
  // Handle structured tool responses (e.g., Bash tool with stdout/stderr)
  if (typeof toolResponse === 'object' && toolResponse !== null) {
    const response = toolResponse as Record<string, unknown>;

    // Extract meaningful content from common response formats
    // Bash tool format: { stdout, stderr, interrupted, isImage }
    if ('stdout' in response || 'stderr' in response) {
      const stdout = typeof response.stdout === 'string' ? response.stdout : '';
      const stderr = typeof response.stderr === 'string' ? response.stderr : '';
      if (stderr && stdout) {
        return `${stdout}\n\nstderr:\n${stderr}`;
      }
      return stderr || stdout;
    }

    // Read tool format: { content, ... } or just text content
    if ('content' in response && typeof response.content === 'string') {
      return response.content;
    }

    // Text field common in many responses
    if ('text' in response && typeof response.text === 'string') {
      return response.text;
    }

    // Result field
    if ('result' in response && typeof response.result === 'string') {
      return response.result;
    }

    // Output field
    if ('output' in response && typeof response.output === 'string') {
      return response.output;
    }

    // Diff payloads for write/edit tools
    if ('diff' in response && typeof response.diff === 'string') {
      return response.diff;
    }
    if ('patch' in response && typeof response.patch === 'string') {
      return response.patch;
    }
    if ('unified_diff' in response && typeof response.unified_diff === 'string') {
      return response.unified_diff;
    }

    // Error responses
    if ('error' in response && typeof response.error === 'string') {
      return `Error: ${response.error}`;
    }

    // Message field
    if ('message' in response && typeof response.message === 'string') {
      return response.message;
    }
  }

  // Fallback to JSON for truly complex objects
  try {
    return JSON.stringify(toolResponse, null, 2);
  } catch {
    return String(toolResponse);
  }
}

function extractPreview(inputs: TurnInput[]): string {
  for (const input of inputs) {
    if (input.type === 'text' && input.text) {
      const trimmed = input.text.trim();
      if (trimmed) {
        return trimmed;
      }
    }
  }
  return '';
}

// Only surface file change tools to match the Codex UI experience.
function isSilentTool(toolName: string, _toolInput?: Record<string, unknown>): boolean {
  const normalized = toolName.toLowerCase();
  const visibleTools = new Set([
    'write',
    'edit',
    'notebookedit',
    'str_replace_editor',
  ]);
  return !visibleTools.has(normalized);
}

function inferToolKind(toolName: string): ToolKind {
  const normalized = toolName.toLowerCase();
  // File modification tools
  if (
    normalized === 'write' ||
    normalized === 'edit' ||
    normalized === 'notebookedit' ||
    normalized === 'str_replace_editor'
  ) {
    return 'fileChange';
  }
  // Everything else (bash) is a command execution
  return 'commandExecution';
}

function inferFileChangeKind(toolName: string): FileChange['kind'] {
  const normalized = toolName.toLowerCase();
  if (normalized.includes('write')) {
    return 'add';
  }
  if (normalized.includes('delete') || normalized.includes('remove')) {
    return 'delete';
  }
  return 'modify';
}

/**
 * Generate a unified diff from old_string and new_string
 * This is used for Edit tool calls to show the changes
 */
function generateUnifiedDiff(
  filePath: string,
  oldString: string,
  newString: string,
): string {
  // If this is a new file (no old content), show the new content
  if (!oldString && newString) {
    const newLines = newString.split('\n');
    const header = `--- /dev/null\n+++ ${filePath}\n@@ -0,0 +1,${newLines.length} @@\n`;
    return header + newLines.map((line) => `+${line}`).join('\n');
  }

  // If deleting content, show the removed content
  if (oldString && !newString) {
    const oldLines = oldString.split('\n');
    const header = `--- ${filePath}\n+++ /dev/null\n@@ -1,${oldLines.length} +0,0 @@\n`;
    return header + oldLines.map((line) => `-${line}`).join('\n');
  }

  // For modifications, generate a unified diff
  const oldLines = oldString.split('\n');
  const newLines = newString.split('\n');

  // Simple unified diff format
  const header = `--- ${filePath}\n+++ ${filePath}\n@@ -1,${oldLines.length} +1,${newLines.length} @@\n`;
  const oldPart = oldLines.map((line) => `-${line}`).join('\n');
  const newPart = newLines.map((line) => `+${line}`).join('\n');

  return header + oldPart + '\n' + newPart;
}

/**
 * Extract diff content from Edit tool input
 */
function extractDiffFromInput(
  toolName: string,
  toolInput: Record<string, unknown>,
  filePath: string,
): string {
  const normalized = toolName.toLowerCase();

  // Edit tool: has old_string and new_string
  if (normalized === 'edit' || normalized === 'str_replace_editor') {
    const oldString = typeof toolInput.old_string === 'string' ? toolInput.old_string : '';
    const newString = typeof toolInput.new_string === 'string' ? toolInput.new_string : '';

    if (oldString || newString) {
      return generateUnifiedDiff(filePath, oldString, newString);
    }
  }

  // Write tool: new file content
  if (normalized === 'write') {
    const content = typeof toolInput.content === 'string' ? toolInput.content : '';
    if (content) {
      return generateUnifiedDiff(filePath, '', content);
    }
  }

  // NotebookEdit tool: cell source
  if (normalized === 'notebookedit') {
    const newSource = typeof toolInput.new_source === 'string' ? toolInput.new_source : '';
    const editMode = typeof toolInput.edit_mode === 'string' ? toolInput.edit_mode : 'replace';
    if (newSource) {
      if (editMode === 'insert') {
        return generateUnifiedDiff(filePath, '', newSource);
      }
      if (editMode === 'delete') {
        return generateUnifiedDiff(filePath, newSource, '');
      }
      // replace mode - show as new content since we don't have old
      return generateUnifiedDiff(filePath, '', newSource);
    }
  }

  return '';
}

function formatCommand(toolName: string, toolInput: Record<string, unknown>): string {
  const normalized = toolName.toLowerCase();
  // For bash, show the actual command
  if (normalized === 'bash') {
    const command = toolInput.command;
    if (typeof command === 'string') {
      return command;
    }
    // Check for description field (Claude SDK sometimes includes this)
    const description = toolInput.description;
    if (typeof description === 'string') {
      return description;
    }
  }
  return toolName;
}

export function buildToolItem(
  toolName: string,
  toolInput: Record<string, unknown>,
  cwd: string,
  itemId: string,
): { item: StoredItem; outputMethod: ToolOutputMethod } {
  const toolKind = inferToolKind(toolName);
  if (toolKind === 'fileChange') {
    const filePath =
      typeof toolInput.file_path === 'string'
        ? toolInput.file_path
        : typeof toolInput.path === 'string'
          ? toolInput.path
          : typeof toolInput.filePath === 'string'
            ? toolInput.filePath
            : typeof toolInput.notebook_path === 'string'
              ? toolInput.notebook_path
              : '';
    // Generate diff from tool input (old_string/new_string for Edit, content for Write)
    const diff = extractDiffFromInput(toolName, toolInput, filePath);
    const changes = filePath
      ? [{ path: filePath, kind: inferFileChangeKind(toolName), diff: diff || undefined }]
      : [];
    return {
      item: {
        id: itemId,
        type: 'fileChange',
        cwd,
        status: 'running',
        changes,
        diff,
      },
      outputMethod: 'item/fileChange/outputDelta',
    };
  }

  return {
    item: {
      id: itemId,
      type: 'commandExecution',
      cwd,
      status: 'running',
      command: formatCommand(toolName, toolInput),
      aggregatedOutput: '',
    },
    outputMethod: 'item/commandExecution/outputDelta',
  };
}

type ContentBlock = { type: string; text?: string; thinking?: string };

function extractAssistantText(message: SDKAssistantMessage['message']): string {
  const blocks = Array.isArray(message.content) ? message.content : [];
  return blocks
    .filter((block: ContentBlock) => block.type === 'text')
    .map((block: ContentBlock) => block.text ?? '')
    .join('')
    .trim();
}

function extractAssistantThinking(message: SDKAssistantMessage['message']): string {
  const blocks = Array.isArray(message.content) ? message.content : [];
  return blocks
    .filter((block: ContentBlock) => block.type === 'thinking')
    .map((block: ContentBlock) => block.thinking ?? '')
    .join('')
    .trim();
}

function isStreamEvent(
  message: SDKMessage,
): message is SDKPartialAssistantMessage {
  return message.type === 'stream_event';
}

function isAssistantMessage(
  message: SDKMessage,
): message is SDKAssistantMessage {
  return message.type === 'assistant';
}

function isResultMessage(message: SDKMessage): message is SDKResultMessage {
  return message.type === 'result';
}

/**
 * Manages Claude Agent SDK sessions and translates events to Codex format
 */
export class ClaudeSession {
  private rpc: JsonRpcHandler;
  private sessions: Map<string, SessionState> = new Map();
  private threads: Map<string, StoredThread> = new Map();
  private approvalRequestId: number = 1;
  private dataPath: string | null = null;
  private loadPromise: Promise<void> | null = null;
  private workspaceId: string;
  private claudeCodePath: string | null;
  private turnContexts: Map<string, TurnContext> = new Map();
  private sessionUsage: SessionUsage = {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
    totalCostUsd: 0,
    turnCount: 0,
    startedAt: Date.now(),
  };
  private threadTokenUsage: Map<string, ThreadTokenUsage> = new Map();

  constructor(rpc: JsonRpcHandler, config: ClaudeSessionConfig) {
    this.rpc = rpc;
    this.workspaceId = config.workspaceId;
    this.claudeCodePath = config.claudeCodePath ?? null;
    if (config.dataDir) {
      this.dataPath = path.join(
        config.dataDir,
        'claude-threads',
        `${config.workspaceId}.json`,
      );
    }
  }

  /**
   * Get current rate limits snapshot for the session
   */
  getRateLimits(): Record<string, unknown> {
    const usage = this.sessionUsage;
    const totalTokens = usage.inputTokens + usage.outputTokens;
    // Use a reasonable session budget for percentage calculation (1M tokens)
    const sessionBudget = 1_000_000;
    const usedPercent = Math.min((totalTokens / sessionBudget) * 100, 100);

    return {
      primary: {
        usedPercent,
        windowDurationMins: null,
        resetsAt: null,
      },
      secondary: null,
      credits: {
        hasCredits: true,
        unlimited: false,
        balance: `$${usage.totalCostUsd.toFixed(4)}`,
      },
      planType: 'claude-api',
      // Additional Claude-specific stats
      usage: {
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        cacheReadTokens: usage.cacheReadTokens,
        cacheCreationTokens: usage.cacheCreationTokens,
        totalTokens,
        totalCostUsd: usage.totalCostUsd,
        turnCount: usage.turnCount,
        sessionDurationMs: Date.now() - usage.startedAt,
      },
    };
  }

  /**
   * Emit rate limits update notification
   */
  private emitRateLimitsUpdate(): void {
    this.rpc.notify('account/rateLimits/updated', {
      rateLimits: this.getRateLimits(),
    });
  }

  private async ensureLoaded(): Promise<void> {
    if (!this.loadPromise) {
      this.loadPromise = this.loadThreads();
    }
    return this.loadPromise;
  }

  private async loadThreads(): Promise<void> {
    if (!this.dataPath) {
      return;
    }
    try {
      const raw = await fs.readFile(this.dataPath, 'utf8');
      const parsed = JSON.parse(raw) as { threads?: StoredThread[] } | null;
      const threads = Array.isArray(parsed?.threads) ? parsed?.threads : [];
      threads.forEach((thread) => {
        if (!thread?.id || !thread?.cwd) {
          return;
        }
        this.threads.set(thread.id, thread);
        this.sessions.set(thread.id, {
          sessionId: thread.sessionId ?? null,
          threadId: thread.id,
          cwd: thread.cwd,
          initialized: true,
        });
      });
    } catch {
      // Ignore missing or malformed thread history.
    }
  }

  private async persistThreads(): Promise<void> {
    if (!this.dataPath) {
      return;
    }
    const dir = path.dirname(this.dataPath);
    await fs.mkdir(dir, { recursive: true });
    const payload = {
      threads: Array.from(this.threads.values()),
    };
    await fs.writeFile(this.dataPath, JSON.stringify(payload, null, 2));
  }

  private getThread(threadId: string, cwd?: string): StoredThread {
    let thread = this.threads.get(threadId) ?? null;
    if (!thread) {
      thread = {
        id: threadId,
        cwd: cwd ?? process.cwd(),
        createdAt: Date.now(),
        updatedAt: Date.now(),
        preview: '',
        sessionId: null,
        turns: [],
      };
      this.threads.set(threadId, thread);
    }
    return thread;
  }

  private getTurn(thread: StoredThread, turnId: string): StoredTurn {
    let turn = thread.turns.find((item) => item.id === turnId) ?? null;
    if (!turn) {
      turn = { id: turnId, createdAt: Date.now(), items: [] };
      thread.turns.push(turn);
    }
    return turn;
  }

  private touchThread(thread: StoredThread): void {
    thread.updatedAt = Date.now();
  }

  private upsertAgentMessage(
    thread: StoredThread,
    turn: StoredTurn,
    itemId: string,
    delta: string,
    isFinal: boolean,
  ): void {
    type AgentMessageItem = Extract<StoredItem, { type: 'agentMessage' }>;
    let item = turn.items.find(
      (entry): entry is AgentMessageItem => entry.id === itemId && entry.type === 'agentMessage',
    );
    if (!item) {
      item = { id: itemId, type: 'agentMessage', text: '' };
      turn.items.push(item);
    }
    item.text = isFinal ? delta : `${item.text}${delta}`;
  }

  private upsertReasoningMessage(
    _thread: StoredThread,
    turn: StoredTurn,
    itemId: string,
    delta: string,
    isFinal: boolean,
  ): void {
    type ReasoningItem = Extract<StoredItem, { type: 'reasoning' }>;
    let item = turn.items.find(
      (entry): entry is ReasoningItem => entry.id === itemId && entry.type === 'reasoning',
    );
    if (!item) {
      item = { id: itemId, type: 'reasoning', content: '' };
      turn.items.push(item);
    }
    const current = item.content ?? '';
    item.content = isFinal ? delta : `${current}${delta}`;
  }

  private upsertToolOutput(
    turn: StoredTurn,
    itemId: string,
    delta: string,
    method: ToolOutputMethod,
  ): void {
    type CommandItem = Extract<StoredItem, { type: 'commandExecution' }>;
    type FileChangeItem = Extract<StoredItem, { type: 'fileChange' }>;
    if (method === 'item/fileChange/outputDelta') {
      const item = turn.items.find(
        (entry): entry is FileChangeItem => entry.id === itemId && entry.type === 'fileChange',
      );
      if (!item) {
        return;
      }
      const current = item.diff ?? '';
      item.diff = `${current}${delta}`;
      if (Array.isArray(item.changes) && item.changes.length > 0) {
        const change = item.changes[0];
        const currentDiff = change.diff ?? '';
        change.diff = `${currentDiff}${delta}`;
      }
      return;
    }
    const item = turn.items.find(
      (entry): entry is CommandItem => entry.id === itemId && entry.type === 'commandExecution',
    );
    if (!item) {
      return;
    }
    const current = item.aggregatedOutput ?? '';
    item.aggregatedOutput = `${current}${delta}`;
  }

  private async buildUserMessage(
    inputs: TurnInput[],
    sessionId: string | null,
  ): Promise<SDKUserMessage> {
    const blocks: Array<Record<string, unknown>> = [];
    for (const input of inputs) {
      if (input.type === 'text' && input.text) {
        const trimmed = input.text.trim();
        if (trimmed) {
          blocks.push({ type: 'text', text: trimmed });
        }
      }
      if (input.type === 'image' && input.url) {
        // Handle data URIs by converting to base64 format
        if (input.url.startsWith('data:')) {
          const match = input.url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            const [, mediaType, base64Data] = match;
            blocks.push({
              type: 'image',
              source: { type: 'base64', data: base64Data, media_type: mediaType },
            });
          } else {
            this.rpc.notify('codex/stderr', {
              message: `Invalid data URI format: ${input.url.substring(0, 50)}...`,
            });
          }
        } else {
          // Regular HTTP/HTTPS URLs
          blocks.push({
            type: 'image',
            source: { type: 'url', url: input.url },
          });
        }
      }
      if (input.type === 'localImage' && input.path) {
        try {
          const data = await fs.readFile(input.path);
          const base64 = data.toString('base64');
          const ext = path.extname(input.path).toLowerCase();
          const mediaType =
            ext === '.jpg' || ext === '.jpeg'
              ? 'image/jpeg'
              : ext === '.gif'
                ? 'image/gif'
                : ext === '.webp'
                  ? 'image/webp'
                  : 'image/png';
          blocks.push({
            type: 'image',
            source: { type: 'base64', data: base64, media_type: mediaType },
          });
        } catch (error) {
          this.rpc.notify('codex/stderr', {
            message: `Failed to read image at ${input.path}: ${
              error instanceof Error ? error.message : String(error)
            }`,
          });
        }
      }
    }

    return {
      type: 'user',
      session_id: sessionId ?? '',
      parent_tool_use_id: null,
      message: {
        role: 'user',
        content: blocks.length ? blocks : [{ type: 'text', text: '' }],
      },
    };
  }

  private async buildPromptInput(
    inputs: TurnInput[],
    sessionId: string | null,
  ): Promise<string | AsyncIterable<SDKUserMessage>> {
    const message = await this.buildUserMessage(inputs, sessionId);
    return (async function* () {
      yield message;
    })();
  }

  /**
   * Handle tool approval - called by Claude SDK's canUseTool callback
   * Sends a JSON-RPC request to the client and waits for response
   */
  async requestToolApproval(
    threadId: string,
    turnId: string,
    toolName: string,
    args: Record<string, unknown>,
    details: {
      toolUseId?: string;
      blockedPath?: string;
      decisionReason?: string;
      suggestions?: unknown;
    },
  ): Promise<PermissionResult> {
    const requestId = this.approvalRequestId++;

    try {
      const response = await this.rpc.request('codex/requestApproval/tool', {
        threadId,
        turnId,
        tool: toolName,
        args,
        blockedPath: details.blockedPath,
        decisionReason: details.decisionReason,
        toolUseId: details.toolUseId,
        suggestions: details.suggestions,
        requestId,
      });

      return mapDecisionToPermissionResult(response, details.toolUseId, args);
    } catch {
      return {
        behavior: 'deny',
        message: 'Tool approval request failed.',
        toolUseID: details.toolUseId,
      };
    }
  }

  /**
   * Start a new thread (creates a new Claude session)
   */
  async startThread(cwd: string): Promise<{ threadId: string; thread: StoredThread }> {
    await this.ensureLoaded();
    const threadId = randomUUID();
    const thread = this.getThread(threadId, cwd);
    this.sessions.set(threadId, {
      sessionId: thread.sessionId ?? null,
      threadId,
      cwd,
      initialized: true,
    });
    await this.persistThreads();

    return { threadId, thread };
  }

  /**
   * Resume an existing thread
   */
  async resumeThread(
    threadId: string,
  ): Promise<{ threadId: string; thread: StoredThread }> {
    await this.ensureLoaded();
    const thread = this.getThread(threadId);
    this.sessions.set(threadId, {
      sessionId: thread.sessionId ?? null,
      threadId,
      cwd: thread.cwd,
      initialized: true,
    });

    return { threadId, thread };
  }

  async listThreads(
    cursor: string | null,
    limit: number,
  ): Promise<{ data: StoredThread[]; nextCursor: string | null }> {
    await this.ensureLoaded();
    const threads = Array.from(this.threads.values()).sort(
      (a, b) => b.updatedAt - a.updatedAt,
    );
    const start = cursor ? Number(cursor) : 0;
    const slice = threads.slice(start, start + limit);
    const nextCursor = start + slice.length < threads.length ? `${start + slice.length}` : null;
    return { data: slice, nextCursor };
  }

  async archiveThread(threadId: string): Promise<{ success: boolean }> {
    await this.ensureLoaded();
    this.threads.delete(threadId);
    this.sessions.delete(threadId);
    await this.persistThreads();
    return { success: true };
  }

  /**
   * Start a new turn (send message to Claude)
   */
  async startTurn(params: TurnStartParams): Promise<{ turn: { id: string } }> {
    await this.ensureLoaded();
    const session = this.sessions.get(params.threadId);
    if (!session) {
      throw new Error('Thread not found');
    }

    const turnId = randomUUID();
    const turnContext: TurnContext = {
      threadId: params.threadId,
      turnId,
      reasoningItemId: null,
      reasoningBlockIndex: null,
      toolOutputMethods: new Map(),
      abortController: new AbortController(),
      activeQuery: null,
      model: params.model ?? null,
    };
    this.turnContexts.set(turnId, turnContext);

    const thread = this.getThread(params.threadId, params.cwd);
    const turn = this.getTurn(thread, turnId);
    const userItem: StoredItem = {
      id: randomUUID(),
      type: 'userMessage',
      content: params.input,
    };
    turn.items.push(userItem);
    const preview = extractPreview(params.input);
    if (!thread.preview && preview) {
      thread.preview = preview;
    }
    this.touchThread(thread);
    await this.persistThreads();

    // Emit turn started
    this.rpc.notify('turn/started', {
      threadId: params.threadId,
      turnId,
    });

    // Build prompt from input
    const prompt = await this.buildPromptInput(params.input, session.sessionId ?? null);

    // Start processing in background
    this.processWithClaude(session, thread, turn, turnContext, prompt, params).catch((error) => {
      const payload = buildTurnErrorEvent(
        params.threadId,
        turnId,
        error instanceof Error ? error.message : String(error),
      );
      this.rpc.notify(payload.method, payload.params);
    });

    return { turn: { id: turnId } };
  }

  /**
   * Interrupt a running turn
   */
  async interruptTurn(threadId: string, turnId: string): Promise<void> {
    const context = this.turnContexts.get(turnId);
    if (!context || context.threadId !== threadId) {
      return;
    }
    context.abortController.abort();
    if (context.activeQuery?.interrupt) {
      await context.activeQuery.interrupt().catch(() => {});
    }
    this.rpc.notify('turn/completed', {
      threadId: context.threadId,
      turnId,
      interrupted: true,
    });
  }

  private async processWithClaude(
    session: SessionState,
    thread: StoredThread,
    turn: StoredTurn,
    context: TurnContext,
    prompt: string | AsyncIterable<SDKUserMessage>,
    params: TurnStartParams,
  ): Promise<void> {
    const { turnId, threadId } = context;
    try {
      const { query } = await import('@anthropic-ai/claude-agent-sdk');
      const approvalPolicy = params.approvalPolicy ?? 'on-request';
      const bypass = approvalPolicy === 'never';
      const sandboxPolicy = params.sandboxPolicy;
      const workspaceCwd = params.cwd || session.cwd;
      const toolOutputMethods = context.toolOutputMethods;

      // Helper to check if a path is within allowed roots
      const isPathAllowed = (filePath: string): boolean => {
        if (!sandboxPolicy || sandboxPolicy.type === 'dangerFullAccess') {
          return true;
        }
        const absolutePath = path.isAbsolute(filePath)
          ? path.normalize(filePath)
          : path.normalize(path.join(workspaceCwd, filePath));
        // Check workspace cwd
        if (absolutePath.startsWith(path.normalize(workspaceCwd) + path.sep) ||
            absolutePath === path.normalize(workspaceCwd)) {
          return true;
        }
        // Check additional writable roots
        if (sandboxPolicy.writableRoots) {
          for (const root of sandboxPolicy.writableRoots) {
            const normalizedRoot = path.normalize(root);
            if (absolutePath.startsWith(normalizedRoot + path.sep) ||
                absolutePath === normalizedRoot) {
              return true;
            }
          }
        }
        return false;
      };

      // Helper to check sandbox policy violations
      const checkSandboxPolicy = (
        toolName: string,
        input: Record<string, unknown>,
      ): { allowed: boolean; reason?: string } => {
        if (!sandboxPolicy || sandboxPolicy.type === 'dangerFullAccess') {
          return { allowed: true };
        }

        const normalizedTool = toolName.toLowerCase();
        const writeTools = new Set([
          'write',
          'edit',
          'notebookedit',
          'str_replace_editor',
        ]);
        const isWriteTool = writeTools.has(normalizedTool);

        const extractFilePath = (): string | undefined => {
          const candidates = [
            input.file_path,
            input.notebook_path,
            input.path,
            input.filePath,
          ];
          for (const candidate of candidates) {
            if (typeof candidate === 'string' && candidate.trim()) {
              return candidate;
            }
          }
          return undefined;
        };

        if (sandboxPolicy.type === 'readOnly') {
          // In read-only mode, deny all write tools and Bash
          if (isWriteTool) {
            return {
              allowed: false,
              reason: `Tool "${toolName}" denied: sandbox policy is read-only`,
            };
          }
          if (normalizedTool === 'bash') {
            return {
              allowed: false,
              reason: 'Bash commands denied: sandbox policy is read-only',
            };
          }
        }

        if (sandboxPolicy.type === 'workspaceWrite') {
          // Check file paths for write tools
          if (isWriteTool) {
            const filePath = extractFilePath();
            if (filePath && !isPathAllowed(filePath)) {
              return {
                allowed: false,
                reason: `Tool "${toolName}" denied: path "${filePath}" is outside allowed workspace roots`,
              };
            }
          }
          // For Bash, we allow it but note it may write outside workspace
          // The approval flow will still apply if not bypassed
        }

        return { allowed: true };
      };

      const canUseTool = async (
        toolName: string,
        input: Record<string, unknown>,
        options: {
          blockedPath?: string;
          decisionReason?: string;
          suggestions?: unknown;
          toolUseID: string;
        },
      ): Promise<PermissionResult> => {
        // Check sandbox policy first
        const sandboxCheck = checkSandboxPolicy(toolName, input);
        if (!sandboxCheck.allowed) {
          return {
            behavior: 'deny',
            toolUseID: options.toolUseID,
            message: sandboxCheck.reason ?? 'Operation denied by sandbox policy',
          };
        }

        if (bypass) {
          return { behavior: 'allow', toolUseID: options.toolUseID };
        }
        return this.requestToolApproval(threadId, turnId, toolName, input, {
          toolUseId: options.toolUseID,
          blockedPath: options.blockedPath,
          decisionReason: options.decisionReason,
          suggestions: options.suggestions,
        });
      };

      const hooks = {
        PreToolUse: [
          {
            hooks: [async (input: HookInput, toolUseID: string | undefined) => {
              if (input.hook_event_name !== 'PreToolUse') {
                return { continue: true };
              }
              const toolUseId = toolUseID ?? input.tool_use_id;
              const toolInput = input.tool_input as Record<string, unknown>;
              if (!threadId || !turnId) {
                return { continue: true };
              }
              if (toolUseId && toolOutputMethods.has(toolUseId)) {
                return { continue: true };
              }
              // Skip emitting events for silent tools (read-only/query tools)
              if (isSilentTool(input.tool_name, toolInput)) {
                return { continue: true };
              }
              const itemId = toolUseId ?? randomUUID();
              const toolCwd =
                typeof input.cwd === 'string' && input.cwd ? input.cwd : thread.cwd;
              const { item, outputMethod } = buildToolItem(
                input.tool_name,
                toolInput,
                toolCwd,
                itemId,
              );
              toolOutputMethods.set(itemId, outputMethod);
              turn.items.push(item);
              this.touchThread(thread);
              const payload = buildToolStartEvent(
                threadId,
                turnId,
                item,
              );
              this.rpc.notify(payload.method, payload.params);
              return { continue: true };
            }],
          },
        ],
        PostToolUse: [
          {
            hooks: [async (input: HookInput, toolUseID: string | undefined) => {
              if (input.hook_event_name !== 'PostToolUse') {
                return { continue: true };
              }
              if (!threadId || !turnId) {
                return { continue: true };
              }
              // Skip emitting events for silent tools (read-only/query tools)
              if (isSilentTool(input.tool_name, input.tool_input as Record<string, unknown>)) {
                return { continue: true };
              }
              const toolUseId = toolUseID ?? input.tool_use_id;
              if (!toolUseId) {
                return { continue: true };
              }
              const outputMethod =
                toolOutputMethods.get(toolUseId) ??
                'item/commandExecution/outputDelta';
              const delta = formatToolOutput(input.tool_response);
              if (delta) {
                this.upsertToolOutput(turn, toolUseId, delta, outputMethod);
                this.touchThread(thread);
                const payload = buildToolOutputEvent(
                  threadId,
                  turnId,
                  toolUseId,
                  delta,
                  outputMethod,
                );
                this.rpc.notify(payload.method, payload.params);
              }
              this.finalizeToolItem(thread, turn, turnId, toolUseId, 'completed', toolOutputMethods);
              await this.persistThreads();
              return { continue: true };
            }],
          },
        ],
        PostToolUseFailure: [
          {
            hooks: [async (input: HookInput, toolUseID: string | undefined) => {
              if (input.hook_event_name !== 'PostToolUseFailure') {
                return { continue: true };
              }
              if (!threadId || !turnId) {
                return { continue: true };
              }
              // Skip emitting events for silent tools (read-only/query tools)
              if (isSilentTool(input.tool_name, input.tool_input as Record<string, unknown>)) {
                return { continue: true };
              }
              const toolUseId = toolUseID ?? input.tool_use_id;
              if (!toolUseId) {
                return { continue: true };
              }
              const outputMethod =
                toolOutputMethods.get(toolUseId) ??
                'item/commandExecution/outputDelta';
              const delta = formatToolOutput(input.error);
              if (delta) {
                this.upsertToolOutput(turn, toolUseId, delta, outputMethod);
                this.touchThread(thread);
                const payload = buildToolOutputEvent(
                  threadId,
                  turnId,
                  toolUseId,
                  delta,
                  outputMethod,
                );
                this.rpc.notify(payload.method, payload.params);
              }
              this.finalizeToolItem(thread, turn, turnId, toolUseId, 'failed', toolOutputMethods);
              await this.persistThreads();
              return { continue: true };
            }],
          },
        ],
      };

      // Determine maxThinkingTokens from effort parameter
      const effort = params.effort ?? 'default';
      const maxThinkingTokens = EFFORT_TO_THINKING_TOKENS[effort];

      const options = {
        cwd: params.cwd || session.cwd,
        model: params.model,
        permissionMode: bypass ? 'bypassPermissions' : 'default',
        allowDangerouslySkipPermissions: bypass,
        systemPrompt: {
          type: 'preset',
          preset: 'claude_code',
        },
        tools: {
          type: 'preset',
          preset: 'claude_code',
        },
        settingSources: ['user', 'project', 'local'],
        resume: session.sessionId ?? undefined,
        canUseTool,
        hooks,
        includePartialMessages: true,
        persistSession: true,
        abortController: context.abortController,
        stderr: (data: string) => {
          this.rpc.notify('codex/stderr', { message: data.trim() });
        },
        pathToClaudeCodeExecutable: this.claudeCodePath ?? undefined,
        maxThinkingTokens,
      } satisfies Parameters<typeof query>[0]['options'];

      // Start an agent message item
      const itemId = randomUUID();
      this.rpc.notify('item/started', {
        threadId: thread.id,
        turnId,
        item: {
          id: itemId,
          type: 'agentMessage',
        },
      });

      let fullContent = '';

      const queryInstance = query({ prompt, options });
      context.activeQuery = queryInstance;

      for await (const message of queryInstance) {
        if (context.abortController.signal.aborted) {
          break;
        }
        const updated = await this.handleSdkMessage(
          thread,
          turn,
          turnId,
          itemId,
          message,
          context,
        );
        if (updated) {
          fullContent = updated;
        }
        const sessionId = (message as { session_id?: string }).session_id;
        if (sessionId && session.sessionId !== sessionId) {
          session.sessionId = sessionId;
          thread.sessionId = sessionId;
          await this.persistThreads();
        }
      }

      // Complete the item
      this.rpc.notify('item/completed', {
        threadId: thread.id,
        turnId,
        item: {
          id: itemId,
          type: 'agentMessage',
          text: fullContent,
        },
      });

      // Complete the turn
      if (!context.abortController.signal.aborted) {
        this.rpc.notify('turn/completed', {
          threadId: thread.id,
          turnId,
          interrupted: false,
        });
      }

      await this.persistThreads();
    } catch (error) {
      const payload = buildTurnErrorEvent(
        thread.id,
        turnId,
        error instanceof Error ? error.message : 'Claude SDK error',
      );
      this.rpc.notify(payload.method, payload.params);
      await this.persistThreads();
    } finally {
      this.turnContexts.delete(turnId);
      context.activeQuery = null;
    }
  }

  private async handleSdkMessage(
    thread: StoredThread,
    turn: StoredTurn,
    turnId: string,
    itemId: string,
    message: SDKMessage,
    context: TurnContext,
  ): Promise<string | null> {
    if (isStreamEvent(message)) {
      const event = message.event;
      if (event.type === 'content_block_delta') {
        if (event.delta.type === 'text_delta') {
          const delta = event.delta.text ?? '';
          if (delta) {
            // Text deltas are Claude's response text - only add to agent message
            // Thinking/reasoning is handled separately via thinking_delta events
            this.upsertAgentMessage(thread, turn, itemId, delta, false);
            this.touchThread(thread);
          }
        }
        if (event.delta.type === 'thinking_delta') {
          const delta = event.delta.thinking ?? '';
          if (delta) {
            if (!context.reasoningItemId) {
              // Fallback: create reasoning item if we missed content_block_start
              const reasoningId = randomUUID();
              context.reasoningItemId = reasoningId;
              const reasoningItem: StoredItem = {
                id: reasoningId,
                type: 'reasoning',
                summary: '',
                content: '',
              };
              turn.items.push(reasoningItem);
              this.touchThread(thread);
              this.rpc.notify('item/started', {
                threadId: thread.id,
                turnId,
                item: reasoningItem,
              });
            }
            const reasoningId = context.reasoningItemId;
            if (!reasoningId) {
              return null;
            }
            // Emit both summaryTextDelta (for UI preview) and textDelta (for full content)
            this.rpc.notify('item/reasoning/summaryTextDelta', {
              threadId: thread.id,
              turnId,
              itemId: reasoningId,
              delta,
            });
            this.rpc.notify('item/reasoning/textDelta', {
              threadId: thread.id,
              turnId,
              itemId: reasoningId,
              delta,
            });
            this.upsertReasoningMessage(thread, turn, reasoningId, delta, false);
            this.touchThread(thread);
          }
        }
      }
      if (event.type === 'content_block_start') {
        const block = event.content_block as { type: string; id?: string; name?: string; input?: unknown };
        const blockIndex = (event as { index?: number }).index;
        if (block.type === 'thinking') {
          // Start a new reasoning block
          const reasoningId = randomUUID();
          context.reasoningItemId = reasoningId;
          context.reasoningBlockIndex = blockIndex ?? null;
          const reasoningItem: StoredItem = {
            id: reasoningId,
            type: 'reasoning',
            summary: '',
            content: '',
          };
          turn.items.push(reasoningItem);
          this.touchThread(thread);
          this.rpc.notify('item/started', {
            threadId: thread.id,
            turnId,
            item: reasoningItem,
          });
        }
        if (block.type === 'tool_use') {
          const toolName = block.name ?? 'Tool';
          const toolInput = (block.input as Record<string, unknown>) ?? {};
          // Skip emitting events for silent tools (read-only/query tools)
          if (isSilentTool(toolName, toolInput)) {
            return null;
          }
          const toolUseId = block.id ?? randomUUID();
          if (!context.toolOutputMethods.has(toolUseId)) {
            const { item, outputMethod } = buildToolItem(
              toolName,
              toolInput,
              thread.cwd,
              toolUseId,
            );
            context.toolOutputMethods.set(toolUseId, outputMethod);
            turn.items.push(item);
            this.touchThread(thread);
            const payload = buildToolStartEvent(
              thread.id,
              turnId,
              item,
            );
            this.rpc.notify(payload.method, payload.params);
          }
        }
      }
      if (event.type === 'content_block_stop') {
        const blockIndex = (event as { index?: number }).index;
        // Check if this is the end of a thinking block
        if (
          context.reasoningItemId &&
          context.reasoningBlockIndex !== null &&
          blockIndex === context.reasoningBlockIndex
        ) {
          // Find the reasoning item and emit completed
          const reasoningItem = turn.items.find(
            (entry) => entry.id === context.reasoningItemId && entry.type === 'reasoning',
          );
          if (reasoningItem) {
            this.rpc.notify('item/completed', {
              threadId: thread.id,
              turnId,
              item: reasoningItem,
            });
          }
          context.reasoningBlockIndex = null;
        }
      }
      return null;
    }

    if (isAssistantMessage(message)) {
      const assistantText = extractAssistantText(message.message);
      if (assistantText) {
        this.upsertAgentMessage(thread, turn, itemId, assistantText, true);
        this.touchThread(thread);
        return assistantText;
      }
      const reasoning = extractAssistantThinking(message.message);
      if (reasoning) {
        const reasoningId = context.reasoningItemId ?? randomUUID();
        context.reasoningItemId = reasoningId;
        this.upsertReasoningMessage(thread, turn, reasoningId, reasoning, true);
        this.touchThread(thread);
      }
      return null;
    }

    if (isResultMessage(message)) {
      const errors =
        'errors' in message && Array.isArray(message.errors)
          ? message.errors
          : [];
      if (message.subtype && message.subtype.startsWith('error')) {
        const errorText = errors.length
          ? errors.join('\n')
          : 'Claude SDK execution failed.';
        const payload = buildTurnErrorEvent(thread.id, turnId, errorText);
        this.rpc.notify(payload.method, payload.params);
      }

      // Extract usage information from successful results
      if (message.subtype === 'success') {
        const result = message as {
          usage?: {
            input_tokens?: number;
            output_tokens?: number;
            cache_read_input_tokens?: number;
            cache_creation_input_tokens?: number;
          };
          modelUsage?: Record<
            string,
            {
              inputTokens?: number;
              outputTokens?: number;
              cacheReadInputTokens?: number;
              cacheCreationInputTokens?: number;
            }
          >;
          total_cost_usd?: number;
        };

        // Calculate turn usage for this turn
        let turnInputTokens = 0;
        let turnOutputTokens = 0;
        let turnCacheReadTokens = 0;

        // Aggregate usage from modelUsage (more detailed) or fallback to usage
        if (result.modelUsage) {
          for (const modelStats of Object.values(result.modelUsage)) {
            turnInputTokens += modelStats.inputTokens ?? 0;
            turnOutputTokens += modelStats.outputTokens ?? 0;
            turnCacheReadTokens += modelStats.cacheReadInputTokens ?? 0;
            this.sessionUsage.inputTokens += modelStats.inputTokens ?? 0;
            this.sessionUsage.outputTokens += modelStats.outputTokens ?? 0;
            this.sessionUsage.cacheReadTokens += modelStats.cacheReadInputTokens ?? 0;
            this.sessionUsage.cacheCreationTokens += modelStats.cacheCreationInputTokens ?? 0;
          }
        } else if (result.usage) {
          turnInputTokens = result.usage.input_tokens ?? 0;
          turnOutputTokens = result.usage.output_tokens ?? 0;
          turnCacheReadTokens = result.usage.cache_read_input_tokens ?? 0;
          this.sessionUsage.inputTokens += turnInputTokens;
          this.sessionUsage.outputTokens += turnOutputTokens;
          this.sessionUsage.cacheReadTokens += turnCacheReadTokens;
          this.sessionUsage.cacheCreationTokens += result.usage.cache_creation_input_tokens ?? 0;
        }

        if (typeof result.total_cost_usd === 'number') {
          this.sessionUsage.totalCostUsd += result.total_cost_usd;
        }
        this.sessionUsage.turnCount += 1;

        // Update and emit thread token usage
        const turnTotalTokens = turnInputTokens + turnOutputTokens;
        const lastUsage: TokenUsageBreakdown = {
          totalTokens: turnTotalTokens,
          inputTokens: turnInputTokens,
          cachedInputTokens: turnCacheReadTokens,
          outputTokens: turnOutputTokens,
          reasoningOutputTokens: 0, // Claude SDK doesn't separate reasoning tokens
        };

        // Get or create thread usage
        const existing = this.threadTokenUsage.get(thread.id);
        const totalUsage: TokenUsageBreakdown = existing
          ? {
              totalTokens: existing.total.totalTokens + lastUsage.totalTokens,
              inputTokens: existing.total.inputTokens + lastUsage.inputTokens,
              cachedInputTokens: existing.total.cachedInputTokens + lastUsage.cachedInputTokens,
              outputTokens: existing.total.outputTokens + lastUsage.outputTokens,
              reasoningOutputTokens: existing.total.reasoningOutputTokens + lastUsage.reasoningOutputTokens,
            }
          : lastUsage;

        const threadUsage: ThreadTokenUsage = {
          total: totalUsage,
          last: lastUsage,
          modelContextWindow: getModelContextWindow(context.model ?? undefined),
        };

        this.threadTokenUsage.set(thread.id, threadUsage);

        // Emit thread token usage update
        this.rpc.notify('thread/tokenUsage/updated', {
          threadId: thread.id,
          tokenUsage: threadUsage,
        });

        // Emit updated rate limits
        this.emitRateLimitsUpdate();
      }

      return null;
    }

    return null;
  }

  private finalizeToolItem(
    thread: StoredThread,
    turn: StoredTurn,
    turnId: string,
    itemId: string,
    status: string,
    toolOutputMethods: Map<string, ToolOutputMethod>,
  ): void {
    const item = turn.items.find(
      (entry) =>
        entry.id === itemId &&
        (entry.type === 'commandExecution' || entry.type === 'fileChange'),
    ) as
      | Extract<StoredItem, { type: 'commandExecution' | 'fileChange' }>
      | undefined;
    if (!item) {
      return;
    }
    item.status = status;
    toolOutputMethods.delete(itemId);
    this.touchThread(thread);
    this.rpc.notify('item/completed', {
      threadId: thread.id,
      turnId,
      item,
    });
  }
}
