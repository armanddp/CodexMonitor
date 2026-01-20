use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use crate::backend::app_server::WorkspaceSession;
use crate::backend::provider::AgentProvider;
use crate::types::WorkspaceEntry;

/// CodexProvider wraps the existing WorkspaceSession to implement AgentProvider
///
/// This provides backward compatibility with the existing Codex integration
/// while enabling the provider abstraction pattern for Claude support.
pub struct CodexProvider {
    session: Arc<WorkspaceSession>,
}

impl CodexProvider {
    /// Create a new CodexProvider wrapping an existing WorkspaceSession
    pub fn new(session: Arc<WorkspaceSession>) -> Self {
        Self { session }
    }
}

#[async_trait]
impl AgentProvider for CodexProvider {
    fn workspace_entry(&self) -> &WorkspaceEntry {
        &self.session.entry
    }

    async fn send_request(&self, method: &str, params: Value) -> Result<Value, String> {
        self.session.send_request(method, params).await
    }

    async fn send_response(&self, id: u64, result: Value) -> Result<(), String> {
        self.session.send_response(id, result).await
    }

    async fn shutdown(&self) -> Result<(), String> {
        let mut child = self.session.child.lock().await;
        child.kill().await.map_err(|e| e.to_string())
    }
}
