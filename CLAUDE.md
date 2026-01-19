# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodexMonitor is a macOS/Linux Tauri desktop app that orchestrates multiple Codex AI agents across local workspaces. The frontend is React 19 + TypeScript + Vite; the backend is Rust (Tauri 2) that spawns `codex app-server` per workspace and streams JSON-RPC events.

## Commands

```bash
# Development
npm install              # Install dependencies
npm run tauri:dev        # Start dev app (runs doctor + Tauri)
npm run dev              # Vite dev server only (localhost:1420)

# Validation (run before committing)
npm run lint             # ESLint check
npm run typecheck        # TypeScript check (no emit)

# Build
npm run tauri:build      # Production bundle (DMG on macOS)
npm run build:appimage   # Linux AppImage

# Troubleshooting
npm run doctor           # Validate environment (CMake, etc.)
npm run doctor:strict    # Fail on missing dependencies
```

## Architecture

**Frontend (`src/`):**
- Feature-sliced architecture in `src/features/` - each feature has `components/`, `hooks/`, and optionally `utils/`
- Components are presentational only (props in, UI out) - no Tauri IPC calls
- Hooks own state, effects, and event wiring
- All Tauri IPC goes through `src/services/tauri.ts`
- Shared types in `src/types.ts`
- CSS organized by area in `src/styles/`

**Backend (`src-tauri/`):**
- `lib.rs` - Tauri commands and app-server client
- `git.rs` - Git operations via libgit2 + `gh` CLI for GitHub issues
- `codex.rs` - Codex app-server JSON-RPC protocol
- `settings.rs` - App settings persistence
- `dictation.rs` - Whisper speech-to-text

**App-Server Protocol:**
- Backend spawns `codex app-server` using the `codex` binary
- JSON-RPC 2.0 over stdio with `initialize`/`initialized` handshake
- Never send requests before initialization completes

## Key Files

- `src/App.tsx` - Composition root; keep orchestration here
- `src/features/app/hooks/useAppServerEvents.ts` - App-server event handling
- `src/features/threads/hooks/useThreads.ts` - Thread lifecycle
- `src/features/git/hooks/useGitStatus.ts` - Git polling and refresh
- `src/utils/threadItems.ts` - Thread item normalization

## Detailed Guidelines

See **AGENTS.md** for comprehensive architecture guidelines, common change patterns, and the app-server protocol flow.
