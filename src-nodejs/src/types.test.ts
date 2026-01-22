import { describe, it, expect } from 'vitest';
import type {
  JsonRpcRequest,
  JsonRpcResponse,
  JsonRpcNotification,
  JsonRpcError,
  InitializeParams,
  ThreadStartParams,
  TurnStartParams,
  TurnInput,
  SandboxPolicy,
} from './types.js';

describe('JSON-RPC Types', () => {
  describe('JsonRpcRequest', () => {
    it('should allow request with method only', () => {
      const request: JsonRpcRequest = { method: 'test/method' };
      expect(request.method).toBe('test/method');
      expect(request.id).toBeUndefined();
      expect(request.params).toBeUndefined();
    });

    it('should allow request with id and params', () => {
      const request: JsonRpcRequest = {
        id: 1,
        method: 'test/method',
        params: { key: 'value' },
      };
      expect(request.id).toBe(1);
      expect(request.params).toEqual({ key: 'value' });
    });
  });

  describe('JsonRpcResponse', () => {
    it('should allow success response', () => {
      const response: JsonRpcResponse = {
        id: 1,
        result: { data: 'test' },
      };
      expect(response.id).toBe(1);
      expect(response.result).toEqual({ data: 'test' });
      expect(response.error).toBeUndefined();
    });

    it('should allow error response', () => {
      const error: JsonRpcError = {
        code: -32600,
        message: 'Invalid Request',
        data: { details: 'missing method' },
      };
      const response: JsonRpcResponse = {
        id: 1,
        error,
      };
      expect(response.error?.code).toBe(-32600);
      expect(response.error?.message).toBe('Invalid Request');
    });
  });

  describe('JsonRpcNotification', () => {
    it('should allow notification with params', () => {
      const notification: JsonRpcNotification = {
        method: 'item/agentMessage/delta',
        params: { text: 'Hello' },
      };
      expect(notification.method).toBe('item/agentMessage/delta');
      expect(notification.params).toEqual({ text: 'Hello' });
    });

    it('should allow notification without params', () => {
      const notification: JsonRpcNotification = {
        method: 'codex/connected',
      };
      expect(notification.params).toBeUndefined();
    });
  });
});

describe('Protocol Types', () => {
  describe('InitializeParams', () => {
    it('should match Codex protocol structure', () => {
      const params: InitializeParams = {
        clientInfo: {
          name: 'codex_monitor',
          title: 'CodexMonitor',
          version: '0.7.5',
        },
      };
      expect(params.clientInfo.name).toBe('codex_monitor');
      expect(params.clientInfo.title).toBe('CodexMonitor');
      expect(params.clientInfo.version).toBe('0.7.5');
    });
  });

  describe('ThreadStartParams', () => {
    it('should support workspace path', () => {
      const params: ThreadStartParams = {
        cwd: '/Users/test/project',
        approvalPolicy: 'on-request',
      };
      expect(params.cwd).toBe('/Users/test/project');
      expect(params.approvalPolicy).toBe('on-request');
    });
  });

  describe('TurnStartParams', () => {
    it('should support full turn configuration', () => {
      const textInput: TurnInput = {
        type: 'text',
        text: 'Fix the bug',
      };
      const imageInput: TurnInput = {
        type: 'localImage',
        path: '/path/to/screenshot.png',
      };
      const sandbox: SandboxPolicy = {
        type: 'workspaceWrite',
        writableRoots: ['/project'],
        networkAccess: true,
      };
      const params: TurnStartParams = {
        threadId: 'thread-123',
        input: [textInput, imageInput],
        cwd: '/project',
        approvalPolicy: 'on-request',
        sandboxPolicy: sandbox,
        model: 'claude-opus-4-5-20251101',
        effort: 'high',
      };
      expect(params.threadId).toBe('thread-123');
      expect(params.input).toHaveLength(2);
      expect(params.input[0].type).toBe('text');
      expect(params.sandboxPolicy?.type).toBe('workspaceWrite');
    });
  });

  describe('TurnInput', () => {
    it('should support text input', () => {
      const input: TurnInput = { type: 'text', text: 'Hello' };
      expect(input.type).toBe('text');
      expect(input.text).toBe('Hello');
    });

    it('should support image URL input', () => {
      const input: TurnInput = { type: 'image', url: 'https://example.com/img.png' };
      expect(input.type).toBe('image');
      expect(input.url).toBe('https://example.com/img.png');
    });

    it('should support local image input', () => {
      const input: TurnInput = { type: 'localImage', path: '/path/to/img.png' };
      expect(input.type).toBe('localImage');
      expect(input.path).toBe('/path/to/img.png');
    });
  });

  describe('SandboxPolicy', () => {
    it('should support full access', () => {
      const policy: SandboxPolicy = { type: 'dangerFullAccess' };
      expect(policy.type).toBe('dangerFullAccess');
    });

    it('should support read-only access', () => {
      const policy: SandboxPolicy = { type: 'readOnly' };
      expect(policy.type).toBe('readOnly');
    });

    it('should support workspace write with network', () => {
      const policy: SandboxPolicy = {
        type: 'workspaceWrite',
        writableRoots: ['/project', '/tmp'],
        networkAccess: true,
      };
      expect(policy.writableRoots).toContain('/project');
      expect(policy.networkAccess).toBe(true);
    });
  });
});
