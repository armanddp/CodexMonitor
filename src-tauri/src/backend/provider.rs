use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::types::WorkspaceEntry;

/// Provider abstraction for agent backends (Codex, Claude, etc.)
///
/// This trait defines the core operations that any agent backend must support.
/// It enables CodexMonitor to work with multiple agent providers through a
/// unified interface.
#[async_trait]
pub trait AgentProvider: Send + Sync {
    /// Returns the workspace entry associated with this provider session
    fn workspace_entry(&self) -> &WorkspaceEntry;

    /// Send a JSON-RPC request and wait for a response
    async fn send_request(&self, method: &str, params: Value) -> Result<Value, String>;

    /// Send a JSON-RPC notification (no response expected)
    async fn send_notification(&self, method: &str, params: Option<Value>) -> Result<(), String>;

    /// Send a JSON-RPC response to a server-initiated request
    async fn send_response(&self, id: u64, result: Value) -> Result<(), String>;

    /// Terminate the provider session
    async fn shutdown(&self) -> Result<(), String>;
}

/// Type alias for a boxed provider session
pub type ProviderSession = Arc<dyn AgentProvider>;
