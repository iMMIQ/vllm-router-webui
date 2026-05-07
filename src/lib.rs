use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};
use tokio::{net::TcpListener, signal};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    core::WorkerType,
    metrics::{self, PrometheusConfig},
    policies::{CacheAwarePolicy, LoadBalancingPolicy, PolicyFactory},
    protocols::worker_spec::{WorkerApiResponse, WorkerConfigRequest, WorkerErrorResponse},
    routers::{
        router_manager::{RouterId, RouterManager},
        RouterFactory, RouterTrait,
    },
    server::{self, AppContext, AppState},
};

#[derive(Debug, Parser, Clone)]
#[command(name = "vllm-router-webui")]
#[command(about = "Single-binary WebUI and control plane for vLLM Router")]
pub struct Cli {
    /// Host address to bind the combined WebUI and router server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to bind the combined WebUI and router server.
    #[arg(long, default_value_t = 30000)]
    pub port: u16,

    /// Initial regular worker URL. Can be specified multiple times.
    #[arg(long = "worker-url")]
    pub worker_urls: Vec<String>,

    /// Default routing policy.
    #[arg(long, default_value = "round_robin")]
    pub policy: String,

    /// Timeout in seconds when validating a worker before adding it through WebUI/API.
    #[arg(long, default_value_t = 30)]
    pub worker_startup_timeout_secs: u64,

    /// Worker health endpoint used by the WebUI control API before adding a worker.
    #[arg(long, default_value = "/health")]
    pub health_endpoint: String,

    /// Maximum request payload size in bytes.
    #[arg(long, default_value_t = 536870912)]
    pub max_payload_size: usize,

    /// Request timeout for proxied worker calls.
    #[arg(long, default_value_t = 120)]
    pub request_timeout_secs: u64,

    /// Maximum concurrent proxied requests.
    #[arg(long, default_value_t = 1024)]
    pub max_concurrent_requests: usize,

    /// Host address for the optional Prometheus metrics server.
    #[arg(long, default_value = "127.0.0.1")]
    pub prometheus_host: String,

    /// Port for the optional Prometheus metrics server. Disabled when omitted.
    #[arg(long)]
    pub prometheus_port: Option<u16>,
}

#[derive(Clone)]
pub struct ControlState {
    app_state: Arc<AppState>,
    client: Client,
    worker_startup_timeout_secs: u64,
    health_endpoint: String,
}

#[derive(Debug, Serialize)]
struct Overview {
    default_policy: String,
    model_policies: HashMap<String, String>,
    worker_counts: HashMap<String, usize>,
    workers: Vec<WorkerView>,
    stats: WorkerStatsView,
    metrics: MetricsSummaryView,
}

#[derive(Debug, Serialize)]
struct WorkerView {
    url: String,
    model_id: String,
    worker_type: String,
    is_healthy: bool,
    load: usize,
    processed_requests: usize,
    is_available: bool,
    priority: u32,
    cost: f32,
    circuit_breaker: CircuitBreakerView,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct WorkerStatsView {
    total_workers: usize,
    healthy_workers: usize,
    total_models: usize,
    total_load: usize,
    regular_workers: usize,
    prefill_workers: usize,
    decode_workers: usize,
}

#[derive(Debug, Serialize)]
struct MetricsSummaryView {
    total_processed_requests: usize,
    unavailable_workers: usize,
    open_circuits: usize,
    half_open_circuits: usize,
    max_worker_load: usize,
}

#[derive(Debug, Serialize)]
struct CircuitBreakerView {
    state: String,
    consecutive_failures: u32,
    consecutive_successes: u32,
    total_failures: u64,
    total_successes: u64,
    time_since_last_failure_ms: Option<u128>,
    time_since_state_change_ms: u128,
}

#[derive(Debug, Deserialize)]
struct RemoveWorkerRequest {
    url: String,
}

#[derive(Debug, Deserialize)]
struct WorkerUrlRequest {
    url: String,
}

#[derive(Debug, Deserialize)]
struct ReplaceWorkerRequest {
    old_url: String,
    new_worker: WorkerConfigRequest,
}

#[derive(Debug, Deserialize)]
struct PolicyUpdateRequest {
    policy: String,
}

#[derive(Debug, Serialize)]
struct PolicyUpdateResponse {
    target: String,
    policy: String,
}

#[derive(Debug, Serialize)]
struct CircuitActionResponse {
    success: bool,
    message: String,
    worker: WorkerView,
}

pub async fn launch(cli: Cli) -> Result<()> {
    let app = build_application(cli.clone()).await?;
    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port)
        .parse()
        .with_context(|| format!("invalid bind address {}:{}", cli.host, cli.port))?;

    info!("Starting vLLM Router WebUI on http://{}", addr);
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

pub async fn build_application(cli: Cli) -> Result<Router> {
    let router_config = build_router_config(&cli)?;
    if let Some(port) = cli.prometheus_port {
        metrics::start_prometheus(PrometheusConfig {
            host: cli.prometheus_host.clone(),
            port,
        });
        info!(
            "Prometheus metrics server enabled on {}:{}",
            cli.prometheus_host, port
        );
    }

    let client = Client::builder()
        .pool_idle_timeout(Some(Duration::from_secs(50)))
        .pool_max_idle_per_host(500)
        .timeout(Duration::from_secs(cli.request_timeout_secs))
        .connect_timeout(Duration::from_secs(10))
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .build()
        .context("failed to create HTTP client")?;

    let context = Arc::new(
        AppContext::new(
            router_config.clone(),
            client.clone(),
            router_config.max_concurrent_requests,
            router_config.rate_limit_tokens_per_second,
            router_config.api_key_validation_urls.clone(),
        )
        .map_err(|e| anyhow!(e))?,
    );

    let router_manager = Arc::new(RouterManager::new(
        router_config.clone(),
        client.clone(),
        context.worker_registry.clone(),
        context.policy_registry.clone(),
    ));

    let regular_router = RouterFactory::create_regular_router(&[], &context)
        .await
        .map_err(|e| anyhow!(e))?;
    router_manager.register_router(
        RouterId::new("http-regular".to_string()),
        Arc::from(regular_router),
    );

    let pd_router = RouterFactory::create_vllm_pd_router(
        &[],
        &[],
        None,
        None,
        None,
        &router_config.policy,
        &context,
    )
    .await
    .map_err(|e| anyhow!(e))?;
    router_manager.register_router(RouterId::new("http-pd".to_string()), Arc::from(pd_router));

    let app_state = Arc::new(AppState {
        router: router_manager.clone() as Arc<dyn RouterTrait>,
        context: context.clone(),
        concurrency_queue_tx: None,
        router_manager: Some(router_manager.clone()),
    });

    let _health_checker = context
        .worker_registry
        .start_health_checker(router_config.health_check.check_interval_secs);

    let control_state = ControlState {
        app_state: app_state.clone(),
        client,
        worker_startup_timeout_secs: cli.worker_startup_timeout_secs,
        health_endpoint: cli.health_endpoint.clone(),
    };

    for worker_url in &cli.worker_urls {
        let config = WorkerConfigRequest {
            url: worker_url.clone(),
            model_id: None,
            priority: None,
            cost: None,
            worker_type: Some("regular".to_string()),
            bootstrap_port: None,
            labels: HashMap::new(),
        };
        if let Err(e) = add_worker_through_manager(&control_state, config).await {
            warn!("Failed to add initial worker {}: {}", worker_url, e.error);
        }
    }

    let upstream_app = server::build_app_with_request_tracing(
        app_state,
        router_config.max_payload_size,
        router_config.request_id_headers.clone().unwrap_or_else(|| {
            vec![
                "x-request-id".to_string(),
                "x-correlation-id".to_string(),
                "x-trace-id".to_string(),
            ]
        }),
        router_config.cors_allowed_origins.clone(),
        false,
        false,
    );

    let control_app = Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/api/overview", get(api_overview))
        .route("/api/workers", post(api_add_worker))
        .route("/api/workers", delete(api_remove_worker))
        .route("/api/workers/replace", post(api_replace_worker))
        .route("/api/workers/circuit/open", post(api_force_open_circuit))
        .route("/api/workers/circuit/reset", post(api_reset_circuit))
        .route("/api/policies/default", post(api_set_default_policy))
        .route(
            "/api/policies/models/{model_id}",
            post(api_set_model_policy),
        )
        .route(
            "/api/policies/models/{model_id}",
            delete(api_clear_model_policy),
        )
        .layer(CorsLayer::permissive())
        .with_state(Arc::new(control_state));

    Ok(upstream_app.merge(control_app))
}

fn build_router_config(cli: &Cli) -> Result<RouterConfig> {
    let mut config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: Vec::new(),
        },
        parse_policy_config(&cli.policy)?,
    );
    config.host = cli.host.clone();
    config.port = cli.port;
    config.enable_igw = true;
    config.max_payload_size = cli.max_payload_size;
    config.request_timeout_secs = cli.request_timeout_secs;
    config.worker_startup_timeout_secs = cli.worker_startup_timeout_secs;
    config.worker_startup_check_interval_secs = 1;
    config.max_concurrent_requests = cli.max_concurrent_requests;
    config.cors_allowed_origins = Vec::new();
    config.health_check.endpoint = cli.health_endpoint.clone();
    Ok(config)
}

fn parse_policy_config(policy: &str) -> Result<PolicyConfig> {
    match policy {
        "random" => Ok(PolicyConfig::Random),
        "round_robin" | "roundrobin" => Ok(PolicyConfig::RoundRobin),
        "cache_aware" | "cacheaware" => Ok(PolicyConfig::CacheAware {
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 120,
            max_tree_size: 67_108_864,
        }),
        "power_of_two" | "poweroftwo" => Ok(PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 5,
        }),
        "consistent_hash" | "consistenthash" => {
            Ok(PolicyConfig::ConsistentHash { virtual_nodes: 160 })
        }
        "rendezvous_hash" | "rendezvoushash" => Ok(PolicyConfig::RendezvousHash),
        other => Err(anyhow!("unsupported policy '{other}'")),
    }
}

fn policy_from_name(
    policy: &str,
) -> Result<Arc<dyn vllm_router_rs::policies::LoadBalancingPolicy>> {
    PolicyFactory::create_by_name(policy).ok_or_else(|| anyhow!("unsupported policy '{policy}'"))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn app_js() -> Response {
    (
        [("content-type", "application/javascript; charset=utf-8")],
        include_str!("../static/app.js"),
    )
        .into_response()
}

async fn api_overview(State(state): State<Arc<ControlState>>) -> Json<Overview> {
    Json(build_overview(&state.app_state))
}

async fn api_add_worker(
    State(state): State<Arc<ControlState>>,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    match add_worker_through_manager(&state, config).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
    }
}

async fn api_remove_worker(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<RemoveWorkerRequest>,
) -> Response {
    match remove_worker_through_manager(&state, &request.url) {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
    }
}

async fn api_replace_worker(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<ReplaceWorkerRequest>,
) -> Response {
    match add_worker_through_manager(&state, request.new_worker).await {
        Ok(_) => match remove_worker_through_manager(&state, &request.old_url) {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
        },
        Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
    }
}

async fn api_force_open_circuit(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<WorkerUrlRequest>,
) -> Response {
    match get_worker_view_by_url(&state.app_state, &request.url, |worker| {
        worker.circuit_breaker().force_open();
    }) {
        Ok(worker) => (
            StatusCode::OK,
            Json(CircuitActionResponse {
                success: true,
                message: format!("Circuit opened for {}", request.url),
                worker,
            }),
        )
            .into_response(),
        Err(error) => (StatusCode::NOT_FOUND, Json(error)).into_response(),
    }
}

async fn api_reset_circuit(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<WorkerUrlRequest>,
) -> Response {
    match get_worker_view_by_url(&state.app_state, &request.url, |worker| {
        worker.circuit_breaker().reset();
    }) {
        Ok(worker) => (
            StatusCode::OK,
            Json(CircuitActionResponse {
                success: true,
                message: format!("Circuit reset for {}", request.url),
                worker,
            }),
        )
            .into_response(),
        Err(error) => (StatusCode::NOT_FOUND, Json(error)).into_response(),
    }
}

async fn api_set_default_policy(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<PolicyUpdateRequest>,
) -> Response {
    match policy_from_name(&request.policy) {
        Ok(policy) => {
            state
                .app_state
                .context
                .policy_registry
                .set_default_policy(policy);
            (
                StatusCode::OK,
                Json(PolicyUpdateResponse {
                    target: "default".to_string(),
                    policy: request.policy,
                }),
            )
                .into_response()
        }
        Err(e) => policy_error(e),
    }
}

async fn api_set_model_policy(
    State(state): State<Arc<ControlState>>,
    Path(model_id): Path<String>,
    Json(request): Json<PolicyUpdateRequest>,
) -> Response {
    match policy_from_name(&request.policy) {
        Ok(policy) => {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    let workers = state
                        .app_state
                        .context
                        .worker_registry
                        .get_by_model_fast(&model_id);
                    cache_aware.init_workers(&workers);
                }
            }
            state
                .app_state
                .context
                .policy_registry
                .set_policy_for_model(&model_id, policy);
            (
                StatusCode::OK,
                Json(PolicyUpdateResponse {
                    target: model_id,
                    policy: request.policy,
                }),
            )
                .into_response()
        }
        Err(e) => policy_error(e),
    }
}

async fn api_clear_model_policy(
    State(state): State<Arc<ControlState>>,
    Path(model_id): Path<String>,
) -> Response {
    state
        .app_state
        .context
        .policy_registry
        .remove_policy_for_model(&model_id);
    (
        StatusCode::OK,
        Json(PolicyUpdateResponse {
            target: model_id,
            policy: "default".to_string(),
        }),
    )
        .into_response()
}

fn policy_error(error: anyhow::Error) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(WorkerErrorResponse {
            error: error.to_string(),
            code: "INVALID_POLICY".to_string(),
        }),
    )
        .into_response()
}

async fn add_worker_through_manager(
    state: &ControlState,
    mut config: WorkerConfigRequest,
) -> Result<WorkerApiResponse, WorkerErrorResponse> {
    wait_for_worker_health(state, &config.url)
        .await
        .map_err(|error| WorkerErrorResponse {
            error,
            code: "WORKER_NOT_READY".to_string(),
        })?;

    if config.worker_type.is_none() {
        config.worker_type = Some("regular".to_string());
    }

    let router_manager =
        state
            .app_state
            .router_manager
            .as_ref()
            .ok_or_else(|| WorkerErrorResponse {
                error: "router manager is not available".to_string(),
                code: "ROUTER_MANAGER_UNAVAILABLE".to_string(),
            })?;

    router_manager.add_worker(config).await
}

fn remove_worker_through_manager(
    state: &ControlState,
    url: &str,
) -> Result<WorkerApiResponse, WorkerErrorResponse> {
    let router_manager =
        state
            .app_state
            .router_manager
            .as_ref()
            .ok_or_else(|| WorkerErrorResponse {
                error: "router manager is not available".to_string(),
                code: "ROUTER_MANAGER_UNAVAILABLE".to_string(),
            })?;

    router_manager.remove_worker_from_registry(url)
}

fn get_worker_view_by_url<F>(
    app_state: &Arc<AppState>,
    url: &str,
    action: F,
) -> Result<WorkerView, WorkerErrorResponse>
where
    F: FnOnce(&Arc<dyn vllm_router_rs::core::Worker>),
{
    let worker = app_state
        .context
        .worker_registry
        .get_by_url(url)
        .ok_or_else(|| WorkerErrorResponse {
            error: format!("Worker {url} not found"),
            code: "WORKER_NOT_FOUND".to_string(),
        })?;
    action(&worker);
    Ok(worker_to_view(worker))
}

async fn wait_for_worker_health(state: &ControlState, worker_url: &str) -> Result<(), String> {
    let health_url = format!(
        "{}{}",
        worker_url.trim_end_matches('/'),
        state.health_endpoint
    );
    let deadline =
        tokio::time::Instant::now() + Duration::from_secs(state.worker_startup_timeout_secs);

    loop {
        match state.client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(format!(
                        "worker {} health check returned {}",
                        worker_url,
                        response.status()
                    ));
                }
            }
            Err(error) => {
                if tokio::time::Instant::now() >= deadline {
                    return Err(format!(
                        "worker {} health check failed: {}",
                        worker_url, error
                    ));
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

fn build_overview(app_state: &Arc<AppState>) -> Overview {
    let stats = app_state.context.worker_registry.stats();
    let workers = app_state
        .context
        .worker_registry
        .get_all()
        .into_iter()
        .map(worker_to_view)
        .collect::<Vec<_>>();

    let metrics = MetricsSummaryView {
        total_processed_requests: workers.iter().map(|worker| worker.processed_requests).sum(),
        unavailable_workers: workers.iter().filter(|worker| !worker.is_available).count(),
        open_circuits: workers
            .iter()
            .filter(|worker| worker.circuit_breaker.state == "open")
            .count(),
        half_open_circuits: workers
            .iter()
            .filter(|worker| worker.circuit_breaker.state == "half_open")
            .count(),
        max_worker_load: workers.iter().map(|worker| worker.load).max().unwrap_or(0),
    };

    Overview {
        default_policy: app_state
            .context
            .policy_registry
            .get_default_policy()
            .name()
            .to_string(),
        model_policies: app_state.context.policy_registry.get_all_mappings(),
        worker_counts: app_state.context.policy_registry.get_worker_counts(),
        workers,
        stats: WorkerStatsView {
            total_workers: stats.total_workers,
            healthy_workers: stats.healthy_workers,
            total_models: stats.total_models,
            total_load: stats.total_load,
            regular_workers: stats.regular_workers,
            prefill_workers: stats.prefill_workers,
            decode_workers: stats.decode_workers,
        },
        metrics,
    }
}

fn worker_to_view(worker: Arc<dyn vllm_router_rs::core::Worker>) -> WorkerView {
    let cb_stats = worker.circuit_breaker().stats();
    WorkerView {
        url: worker.url().to_string(),
        model_id: worker.model_id().to_string(),
        worker_type: match worker.worker_type() {
            WorkerType::Regular => "regular".to_string(),
            WorkerType::Prefill { .. } => "prefill".to_string(),
            WorkerType::Decode => "decode".to_string(),
        },
        is_healthy: worker.is_healthy(),
        load: worker.load(),
        processed_requests: worker.processed_requests(),
        is_available: worker.is_available(),
        priority: worker.priority(),
        cost: worker.cost(),
        circuit_breaker: CircuitBreakerView {
            state: circuit_state_name(cb_stats.state),
            consecutive_failures: cb_stats.consecutive_failures,
            consecutive_successes: cb_stats.consecutive_successes,
            total_failures: cb_stats.total_failures,
            total_successes: cb_stats.total_successes,
            time_since_last_failure_ms: cb_stats
                .time_since_last_failure
                .map(|duration| duration.as_millis()),
            time_since_state_change_ms: cb_stats.time_since_last_state_change.as_millis(),
        },
        metadata: worker.metadata().labels.clone(),
    }
}

fn circuit_state_name(state: vllm_router_rs::core::CircuitState) -> String {
    match state {
        vllm_router_rs::core::CircuitState::Closed => "closed",
        vllm_router_rs::core::CircuitState::Open => "open",
        vllm_router_rs::core::CircuitState::HalfOpen => "half_open",
    }
    .to_string()
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        let mut signal = signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        signal.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
