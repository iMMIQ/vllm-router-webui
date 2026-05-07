use anyhow::Result;
use clap::Parser;
use vllm_router_webui::{launch, Cli};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "vllm_router_webui=info,vllm_router_rs=info".into()),
        )
        .init();

    launch(Cli::parse()).await
}
