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
use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{net::TcpListener, signal};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use vllm_router_rs::{
    config::{PolicyConfig, RouterConfig, RoutingMode},
    core::{CircuitBreakerConfig, HealthConfig, Worker, WorkerType},
    metrics::{self, PrometheusConfig},
    policies::{LoadBalancingPolicy, PolicyFactory},
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

    /// JSON file used to persist WebUI control-plane state. Disabled when omitted.
    #[arg(long)]
    pub state_file: Option<PathBuf>,

    /// Maximum number of config revisions kept in the persisted state file.
    #[arg(long, default_value_t = 50)]
    pub max_revisions: usize,
}

#[derive(Clone)]
pub struct ControlState {
    app_state: Arc<AppState>,
    client: Client,
    worker_startup_timeout_secs: u64,
    health_endpoint: String,
    default_circuit_breaker_config: Arc<RwLock<CircuitBreakerConfig>>,
    default_health_config: Arc<RwLock<HealthConfig>>,
    config_store: Arc<RwLock<ConfigStore>>,
    state_file: Option<PathBuf>,
    max_revisions: usize,
    metric_history: Arc<RwLock<Vec<MetricPoint>>>,
}

#[derive(Debug, Serialize)]
struct Overview {
    default_policy: String,
    model_policies: HashMap<String, String>,
    worker_counts: HashMap<String, usize>,
    workers: Vec<WorkerView>,
    stats: WorkerStatsView,
    metrics: MetricsSummaryView,
    runtime_config: RuntimeConfigView,
    config: ConfigMetadataView,
    trends: Vec<MetricPoint>,
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
    health_check: HealthConfigView,
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
    config: CircuitBreakerConfigView,
}

#[derive(Debug, Serialize)]
struct RuntimeConfigView {
    circuit_breaker: CircuitBreakerConfigView,
    health_check: HealthConfigView,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CircuitBreakerConfigView {
    failure_threshold: u32,
    success_threshold: u32,
    timeout_duration_secs: u64,
    window_duration_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HealthConfigView {
    timeout_secs: u64,
    check_interval_secs: u64,
    endpoint: String,
    failure_threshold: u32,
    success_threshold: u32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyUpdateRequest {
    policy: String,
    cache_threshold: Option<f32>,
    balance_abs_threshold: Option<usize>,
    balance_rel_threshold: Option<f32>,
    eviction_interval_secs: Option<u64>,
    max_tree_size: Option<usize>,
    load_check_interval_secs: Option<u64>,
    virtual_nodes: Option<u32>,
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

#[derive(Debug, Deserialize)]
struct CircuitBreakerConfigUpdateRequest {
    url: Option<String>,
    failure_threshold: Option<u32>,
    success_threshold: Option<u32>,
    timeout_duration_secs: Option<u64>,
    window_duration_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HealthConfigUpdateRequest {
    url: Option<String>,
    timeout_secs: Option<u64>,
    check_interval_secs: Option<u64>,
    endpoint: Option<String>,
    failure_threshold: Option<u32>,
    success_threshold: Option<u32>,
}

#[derive(Debug, Serialize)]
struct RuntimeConfigUpdateResponse {
    success: bool,
    message: String,
    applied_workers: usize,
    runtime_config: RuntimeConfigView,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigStore {
    current: PersistedConfig,
    revisions: Vec<ConfigRevision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedConfig {
    workers: Vec<WorkerConfigRequest>,
    default_policy: PolicyUpdateRequest,
    model_policies: HashMap<String, PolicyUpdateRequest>,
    default_circuit_breaker: CircuitBreakerConfigView,
    default_health: HealthConfigView,
    worker_circuit_breakers: HashMap<String, CircuitBreakerConfigView>,
    worker_health: HashMap<String, HealthConfigView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigRevision {
    id: u64,
    timestamp_secs: u64,
    description: String,
    config: PersistedConfig,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigMetadataView {
    current_revision: Option<u64>,
    revision_count: usize,
    persistence_enabled: bool,
    state_file: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigHistoryView {
    current_revision: Option<u64>,
    revisions: Vec<ConfigRevisionSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigRevisionSummary {
    id: u64,
    timestamp_secs: u64,
    description: String,
    workers: usize,
    model_policies: usize,
}

#[derive(Debug, Deserialize)]
struct RollbackRequest {
    revision_id: u64,
}

#[derive(Debug, Serialize)]
struct ConfigActionResponse {
    success: bool,
    message: String,
    current_revision: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct MetricPoint {
    timestamp_secs: u64,
    total_workers: usize,
    healthy_workers: usize,
    unavailable_workers: usize,
    open_circuits: usize,
    max_worker_load: usize,
    total_processed_requests: usize,
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
    let persisted_store = load_config_store(cli.state_file.as_ref())?;
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

    let initial_config = initial_persisted_config(&cli, &router_config, persisted_store.as_ref())?;
    let control_state = ControlState {
        app_state: app_state.clone(),
        client,
        worker_startup_timeout_secs: cli.worker_startup_timeout_secs,
        health_endpoint: cli.health_endpoint.clone(),
        default_circuit_breaker_config: Arc::new(RwLock::new(
            initial_config.default_circuit_breaker.to_core_config(),
        )),
        default_health_config: Arc::new(RwLock::new(
            initial_config.default_health.to_core_config(),
        )),
        config_store: Arc::new(RwLock::new(
            persisted_store.unwrap_or_else(|| ConfigStore::new(initial_config.clone())),
        )),
        state_file: cli.state_file.clone(),
        max_revisions: cli.max_revisions.max(1),
        metric_history: Arc::new(RwLock::new(Vec::new())),
    };

    apply_default_policy_request(&control_state, &initial_config.default_policy)?;

    for (model_id, policy_request) in &initial_config.model_policies {
        apply_model_policy_request(&control_state, model_id, policy_request)?;
    }

    for config in initial_config.workers.clone() {
        let worker_url = config.url.clone();
        if let Err(e) = add_worker_through_manager(&control_state, config).await {
            warn!(
                "Failed to add persisted or initial worker {}: {}",
                worker_url, e.error
            );
        }
    }

    apply_runtime_snapshot(&control_state, &initial_config);
    persist_revision(&control_state, "initial state")?;

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
        .route(
            "/api/runtime/circuit-breaker",
            post(api_update_circuit_breaker_config),
        )
        .route("/api/runtime/health", post(api_update_health_config))
        .route("/api/config/history", get(api_config_history))
        .route("/api/config/rollback", post(api_config_rollback))
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
    Json(build_overview(&state))
}

async fn api_add_worker(
    State(state): State<Arc<ControlState>>,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    match add_worker_through_manager(&state, config.clone()).await {
        Ok(response) => {
            upsert_persisted_worker(&state, config);
            if let Err(error) = persist_revision(&state, "worker added") {
                return persistence_error(error);
            }
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
    }
}

async fn api_remove_worker(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<RemoveWorkerRequest>,
) -> Response {
    match remove_worker_through_manager(&state, &request.url) {
        Ok(response) => {
            remove_persisted_worker(&state, &request.url);
            if let Err(error) = persist_revision(&state, "worker removed") {
                return persistence_error(error);
            }
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
    }
}

async fn api_replace_worker(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<ReplaceWorkerRequest>,
) -> Response {
    match add_worker_through_manager(&state, request.new_worker.clone()).await {
        Ok(_) => match remove_worker_through_manager(&state, &request.old_url) {
            Ok(response) => {
                remove_persisted_worker(&state, &request.old_url);
                upsert_persisted_worker(&state, request.new_worker);
                if let Err(error) = persist_revision(&state, "worker replaced") {
                    return persistence_error(error);
                }
                (StatusCode::OK, Json(response)).into_response()
            }
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
    match apply_default_policy_request(&state, &request) {
        Ok(policy_config) => {
            {
                let mut store = state.config_store.write().unwrap();
                store.current.default_policy = request;
            }
            if let Err(error) = persist_revision(&state, "default policy updated") {
                return persistence_error(error);
            }
            (
                StatusCode::OK,
                Json(PolicyUpdateResponse {
                    target: "default".to_string(),
                    policy: policy_config.name().to_string(),
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
    match apply_model_policy_request(&state, &model_id, &request) {
        Ok(policy_config) => {
            {
                let mut store = state.config_store.write().unwrap();
                store
                    .current
                    .model_policies
                    .insert(model_id.clone(), request);
            }
            if let Err(error) = persist_revision(&state, "model policy updated") {
                return persistence_error(error);
            }
            (
                StatusCode::OK,
                Json(PolicyUpdateResponse {
                    target: model_id,
                    policy: policy_config.name().to_string(),
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
    {
        let mut store = state.config_store.write().unwrap();
        store.current.model_policies.remove(&model_id);
    }
    if let Err(error) = persist_revision(&state, "model policy cleared") {
        return persistence_error(error);
    }
    (
        StatusCode::OK,
        Json(PolicyUpdateResponse {
            target: model_id,
            policy: "default".to_string(),
        }),
    )
        .into_response()
}

fn persistence_error(error: anyhow::Error) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(WorkerErrorResponse {
            error: error.to_string(),
            code: "PERSISTENCE_ERROR".to_string(),
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

async fn api_update_circuit_breaker_config(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<CircuitBreakerConfigUpdateRequest>,
) -> Response {
    let current = if let Some(url) = request.url.as_deref() {
        state
            .app_state
            .context
            .worker_registry
            .get_by_url(url)
            .map(|worker| worker.circuit_breaker().config())
    } else {
        Some(state.default_circuit_breaker_config.read().unwrap().clone())
    };

    let Some(current) = current else {
        return (
            StatusCode::NOT_FOUND,
            Json(WorkerErrorResponse {
                error: format!(
                    "Worker {} not found",
                    request.url.as_deref().unwrap_or_default()
                ),
                code: "WORKER_NOT_FOUND".to_string(),
            }),
        )
            .into_response();
    };

    let updated = merge_circuit_breaker_config(current, &request);
    let applied_workers = if let Some(url) = request.url.as_deref() {
        match state.app_state.context.worker_registry.get_by_url(url) {
            Some(worker) => {
                worker.circuit_breaker().update_config(updated.clone());
                1
            }
            None => 0,
        }
    } else {
        *state.default_circuit_breaker_config.write().unwrap() = updated.clone();
        let workers = state.app_state.context.worker_registry.get_all();
        for worker in &workers {
            worker.circuit_breaker().update_config(updated.clone());
        }
        workers.len()
    };

    {
        let mut store = state.config_store.write().unwrap();
        let updated_view = CircuitBreakerConfigView::from_config(&updated);
        if let Some(url) = request.url.as_ref() {
            store
                .current
                .worker_circuit_breakers
                .insert(url.clone(), updated_view);
        } else {
            store.current.default_circuit_breaker = updated_view;
            store.current.worker_circuit_breakers.clear();
        }
    }
    if let Err(error) = persist_revision(&state, "circuit breaker config updated") {
        return persistence_error(error);
    }

    (
        StatusCode::OK,
        Json(RuntimeConfigUpdateResponse {
            success: true,
            message: if request.url.is_some() {
                "Circuit breaker config updated for worker".to_string()
            } else {
                "Default circuit breaker config updated".to_string()
            },
            applied_workers,
            runtime_config: runtime_config_view(&state),
        }),
    )
        .into_response()
}

async fn api_update_health_config(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<HealthConfigUpdateRequest>,
) -> Response {
    let current = if let Some(url) = request.url.as_deref() {
        state
            .app_state
            .context
            .worker_registry
            .get_by_url(url)
            .map(|worker| worker.health_config())
    } else {
        Some(state.default_health_config.read().unwrap().clone())
    };

    let Some(current) = current else {
        return (
            StatusCode::NOT_FOUND,
            Json(WorkerErrorResponse {
                error: format!(
                    "Worker {} not found",
                    request.url.as_deref().unwrap_or_default()
                ),
                code: "WORKER_NOT_FOUND".to_string(),
            }),
        )
            .into_response();
    };

    let updated = merge_health_config(current, &request);
    let applied_workers = if let Some(url) = request.url.as_deref() {
        match state.app_state.context.worker_registry.get_by_url(url) {
            Some(worker) => {
                worker.update_health_config(updated.clone());
                1
            }
            None => 0,
        }
    } else {
        *state.default_health_config.write().unwrap() = updated.clone();
        let workers = state.app_state.context.worker_registry.get_all();
        for worker in &workers {
            worker.update_health_config(updated.clone());
        }
        workers.len()
    };

    {
        let mut store = state.config_store.write().unwrap();
        let updated_view = HealthConfigView::from_config(&updated);
        if let Some(url) = request.url.as_ref() {
            store
                .current
                .worker_health
                .insert(url.clone(), updated_view);
        } else {
            store.current.default_health = updated_view;
            store.current.worker_health.clear();
        }
    }
    if let Err(error) = persist_revision(&state, "health config updated") {
        return persistence_error(error);
    }

    (
        StatusCode::OK,
        Json(RuntimeConfigUpdateResponse {
            success: true,
            message: if request.url.is_some() {
                "Health check config updated for worker".to_string()
            } else {
                "Default health check config updated".to_string()
            },
            applied_workers,
            runtime_config: runtime_config_view(&state),
        }),
    )
        .into_response()
}

async fn api_config_history(State(state): State<Arc<ControlState>>) -> Json<ConfigHistoryView> {
    let store = state.config_store.read().unwrap();
    Json(ConfigHistoryView {
        current_revision: store.current_revision_id(),
        revisions: store
            .revisions
            .iter()
            .rev()
            .map(ConfigRevisionSummary::from_revision)
            .collect(),
    })
}

async fn api_config_rollback(
    State(state): State<Arc<ControlState>>,
    Json(request): Json<RollbackRequest>,
) -> Response {
    let target = {
        let store = state.config_store.read().unwrap();
        store
            .revisions
            .iter()
            .find(|revision| revision.id == request.revision_id)
            .cloned()
    };

    let Some(target) = target else {
        return (
            StatusCode::NOT_FOUND,
            Json(WorkerErrorResponse {
                error: format!("Revision {} not found", request.revision_id),
                code: "REVISION_NOT_FOUND".to_string(),
            }),
        )
            .into_response();
    };

    if let Err(error) = apply_persisted_config(&state, &target.config).await {
        return (
            StatusCode::BAD_REQUEST,
            Json(WorkerErrorResponse {
                error: error.to_string(),
                code: "ROLLBACK_FAILED".to_string(),
            }),
        )
            .into_response();
    }

    {
        let mut store = state.config_store.write().unwrap();
        store.current = target.config;
    }

    if let Err(error) = persist_revision(&state, &format!("rollback to {}", request.revision_id)) {
        return persistence_error(error);
    }

    let current_revision = state.config_store.read().unwrap().current_revision_id();
    (
        StatusCode::OK,
        Json(ConfigActionResponse {
            success: true,
            message: format!("Rolled back to revision {}", request.revision_id),
            current_revision,
        }),
    )
        .into_response()
}

async fn add_worker_through_manager(
    state: &ControlState,
    mut config: WorkerConfigRequest,
) -> Result<WorkerApiResponse, WorkerErrorResponse> {
    let worker_url = config.url.clone();

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

    let response = router_manager.add_worker(config).await?;
    if let Some(worker) = state
        .app_state
        .context
        .worker_registry
        .get_by_url(&worker_url)
    {
        apply_runtime_defaults_to_worker(state, &worker);
    }
    Ok(response)
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

fn build_overview(state: &ControlState) -> Overview {
    let app_state = &state.app_state;
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
    record_metric_point(state, stats.total_workers, stats.healthy_workers, &metrics);
    let store = state.config_store.read().unwrap();

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
        runtime_config: runtime_config_view(state),
        config: ConfigMetadataView {
            current_revision: store.current_revision_id(),
            revision_count: store.revisions.len(),
            persistence_enabled: state.state_file.is_some(),
            state_file: state
                .state_file
                .as_ref()
                .map(|path| path.display().to_string()),
        },
        trends: state.metric_history.read().unwrap().clone(),
    }
}

fn record_metric_point(
    state: &ControlState,
    total_workers: usize,
    healthy_workers: usize,
    metrics: &MetricsSummaryView,
) {
    let mut history = state.metric_history.write().unwrap();
    let now = now_secs();
    if history
        .last()
        .is_some_and(|point| point.timestamp_secs == now)
    {
        return;
    }
    history.push(MetricPoint {
        timestamp_secs: now,
        total_workers,
        healthy_workers,
        unavailable_workers: metrics.unavailable_workers,
        open_circuits: metrics.open_circuits,
        max_worker_load: metrics.max_worker_load,
        total_processed_requests: metrics.total_processed_requests,
    });
    if history.len() > 120 {
        let remove_count = history.len() - 120;
        history.drain(0..remove_count);
    }
}

fn worker_to_view(worker: Arc<dyn vllm_router_rs::core::Worker>) -> WorkerView {
    let cb_stats = worker.circuit_breaker().stats();
    let health_config = worker.health_config();
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
            config: CircuitBreakerConfigView::from_config(&cb_stats.config),
        },
        health_check: HealthConfigView::from_config(&health_config),
        metadata: worker.metadata().labels.clone(),
    }
}

fn default_circuit_breaker_config(router_config: &RouterConfig) -> CircuitBreakerConfig {
    let mut config = CircuitBreakerConfig {
        failure_threshold: router_config.circuit_breaker.failure_threshold,
        success_threshold: router_config.circuit_breaker.success_threshold,
        timeout_duration: Duration::from_secs(router_config.circuit_breaker.timeout_duration_secs),
        window_duration: Duration::from_secs(router_config.circuit_breaker.window_duration_secs),
    };
    if router_config.disable_circuit_breaker {
        config.failure_threshold = u32::MAX;
    }
    config
}

fn default_health_config(router_config: &RouterConfig) -> HealthConfig {
    HealthConfig {
        timeout_secs: router_config.health_check.timeout_secs,
        check_interval_secs: router_config.health_check.check_interval_secs,
        endpoint: router_config.health_check.endpoint.clone(),
        failure_threshold: router_config.health_check.failure_threshold,
        success_threshold: router_config.health_check.success_threshold,
    }
}

fn runtime_config_view(state: &ControlState) -> RuntimeConfigView {
    RuntimeConfigView {
        circuit_breaker: CircuitBreakerConfigView::from_config(
            &state.default_circuit_breaker_config.read().unwrap(),
        ),
        health_check: HealthConfigView::from_config(&state.default_health_config.read().unwrap()),
    }
}

impl CircuitBreakerConfigView {
    fn from_config(config: &CircuitBreakerConfig) -> Self {
        Self {
            failure_threshold: config.failure_threshold,
            success_threshold: config.success_threshold,
            timeout_duration_secs: config.timeout_duration.as_secs(),
            window_duration_secs: config.window_duration.as_secs(),
        }
    }
}

impl HealthConfigView {
    fn from_config(config: &HealthConfig) -> Self {
        Self {
            timeout_secs: config.timeout_secs,
            check_interval_secs: config.check_interval_secs,
            endpoint: config.endpoint.clone(),
            failure_threshold: config.failure_threshold,
            success_threshold: config.success_threshold,
        }
    }

    fn to_core_config(&self) -> HealthConfig {
        HealthConfig {
            timeout_secs: self.timeout_secs,
            check_interval_secs: self.check_interval_secs,
            endpoint: self.endpoint.clone(),
            failure_threshold: self.failure_threshold,
            success_threshold: self.success_threshold,
        }
    }
}

impl CircuitBreakerConfigView {
    fn to_core_config(&self) -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: self.failure_threshold,
            success_threshold: self.success_threshold,
            timeout_duration: Duration::from_secs(self.timeout_duration_secs),
            window_duration: Duration::from_secs(self.window_duration_secs),
        }
    }
}

fn merge_circuit_breaker_config(
    mut config: CircuitBreakerConfig,
    request: &CircuitBreakerConfigUpdateRequest,
) -> CircuitBreakerConfig {
    if let Some(failure_threshold) = request.failure_threshold {
        config.failure_threshold = failure_threshold;
    }
    if let Some(success_threshold) = request.success_threshold {
        config.success_threshold = success_threshold;
    }
    if let Some(timeout_duration_secs) = request.timeout_duration_secs {
        config.timeout_duration = Duration::from_secs(timeout_duration_secs);
    }
    if let Some(window_duration_secs) = request.window_duration_secs {
        config.window_duration = Duration::from_secs(window_duration_secs);
    }
    config
}

fn merge_health_config(
    mut config: HealthConfig,
    request: &HealthConfigUpdateRequest,
) -> HealthConfig {
    if let Some(timeout_secs) = request.timeout_secs {
        config.timeout_secs = timeout_secs;
    }
    if let Some(check_interval_secs) = request.check_interval_secs {
        config.check_interval_secs = check_interval_secs;
    }
    if let Some(endpoint) = request.endpoint.as_ref() {
        config.endpoint = endpoint.clone();
    }
    if let Some(failure_threshold) = request.failure_threshold {
        config.failure_threshold = failure_threshold;
    }
    if let Some(success_threshold) = request.success_threshold {
        config.success_threshold = success_threshold;
    }
    config
}

fn apply_runtime_defaults_to_worker(state: &ControlState, worker: &Arc<dyn Worker>) {
    worker
        .circuit_breaker()
        .update_config(state.default_circuit_breaker_config.read().unwrap().clone());
    worker.update_health_config(state.default_health_config.read().unwrap().clone());
}

fn upsert_persisted_worker(state: &ControlState, config: WorkerConfigRequest) {
    let mut store = state.config_store.write().unwrap();
    store
        .current
        .workers
        .retain(|worker| worker.url != config.url);
    store.current.workers.push(config);
}

fn remove_persisted_worker(state: &ControlState, url: &str) {
    let mut store = state.config_store.write().unwrap();
    store.current.workers.retain(|worker| worker.url != url);
    store.current.worker_circuit_breakers.remove(url);
    store.current.worker_health.remove(url);
}

fn policy_from_request(
    request: &PolicyUpdateRequest,
) -> Result<(PolicyConfig, Arc<dyn LoadBalancingPolicy>)> {
    let config = policy_config_from_request(request)?;
    let policy = PolicyFactory::create_from_config(&config);
    Ok((config, policy))
}

fn policy_config_from_request(request: &PolicyUpdateRequest) -> Result<PolicyConfig> {
    let policy = request.policy.to_lowercase();
    match policy.as_str() {
        "random" => Ok(PolicyConfig::Random),
        "round_robin" | "roundrobin" => Ok(PolicyConfig::RoundRobin),
        "cache_aware" | "cacheaware" => Ok(PolicyConfig::CacheAware {
            cache_threshold: request.cache_threshold.unwrap_or(0.3),
            balance_abs_threshold: request.balance_abs_threshold.unwrap_or(64),
            balance_rel_threshold: request.balance_rel_threshold.unwrap_or(1.5),
            eviction_interval_secs: request.eviction_interval_secs.unwrap_or(120),
            max_tree_size: request.max_tree_size.unwrap_or(67_108_864),
        }),
        "power_of_two" | "poweroftwo" => Ok(PolicyConfig::PowerOfTwo {
            load_check_interval_secs: request.load_check_interval_secs.unwrap_or(5),
        }),
        "consistent_hash" | "consistenthash" => Ok(PolicyConfig::ConsistentHash {
            virtual_nodes: request.virtual_nodes.unwrap_or(160),
        }),
        "rendezvous_hash" | "rendezvoushash" => Ok(PolicyConfig::RendezvousHash),
        other => Err(anyhow!("unsupported policy '{other}'")),
    }
}

fn initialize_policy_if_needed(policy: &Arc<dyn LoadBalancingPolicy>, workers: &[Arc<dyn Worker>]) {
    if policy.requires_initialization() {
        policy.init_workers(workers);
    }
}

fn apply_default_policy_request(
    state: &ControlState,
    request: &PolicyUpdateRequest,
) -> Result<PolicyConfig> {
    let (policy_config, policy) = policy_from_request(request)?;
    initialize_policy_if_needed(&policy, &state.app_state.context.worker_registry.get_all());
    state
        .app_state
        .context
        .policy_registry
        .set_default_policy(policy);
    Ok(policy_config)
}

fn apply_model_policy_request(
    state: &ControlState,
    model_id: &str,
    request: &PolicyUpdateRequest,
) -> Result<PolicyConfig> {
    let (policy_config, policy) = policy_from_request(request)?;
    let workers = state
        .app_state
        .context
        .worker_registry
        .get_by_model_fast(model_id);
    initialize_policy_if_needed(&policy, &workers);
    state
        .app_state
        .context
        .policy_registry
        .set_policy_for_model(model_id, policy);
    Ok(policy_config)
}

async fn apply_persisted_config(state: &ControlState, config: &PersistedConfig) -> Result<()> {
    let desired_urls = config
        .workers
        .iter()
        .map(|worker| worker.url.clone())
        .collect::<std::collections::HashSet<_>>();
    let current_workers = state.app_state.context.worker_registry.get_all();

    for worker in current_workers {
        if !desired_urls.contains(worker.url()) {
            remove_worker_through_manager(state, worker.url()).map_err(|error| {
                anyhow!("failed to remove worker {}: {}", worker.url(), error.error)
            })?;
        }
    }

    for worker_config in &config.workers {
        if state
            .app_state
            .context
            .worker_registry
            .get_by_url(&worker_config.url)
            .is_none()
        {
            add_worker_through_manager(state, worker_config.clone())
                .await
                .map_err(|error| {
                    anyhow!(
                        "failed to add worker {}: {}",
                        worker_config.url,
                        error.error
                    )
                })?;
        }
    }

    apply_default_policy_request(state, &config.default_policy)?;
    for model_id in state
        .app_state
        .context
        .policy_registry
        .get_all_mappings()
        .keys()
        .cloned()
        .collect::<Vec<_>>()
    {
        if !config.model_policies.contains_key(&model_id) {
            state
                .app_state
                .context
                .policy_registry
                .remove_policy_for_model(&model_id);
        }
    }
    for (model_id, policy_request) in &config.model_policies {
        apply_model_policy_request(state, model_id, policy_request)?;
    }

    apply_runtime_snapshot(state, config);
    Ok(())
}

fn apply_runtime_snapshot(state: &ControlState, config: &PersistedConfig) {
    let default_cb = config.default_circuit_breaker.to_core_config();
    let default_health = config.default_health.to_core_config();
    *state.default_circuit_breaker_config.write().unwrap() = default_cb.clone();
    *state.default_health_config.write().unwrap() = default_health.clone();

    for worker in state.app_state.context.worker_registry.get_all() {
        let cb = config
            .worker_circuit_breakers
            .get(worker.url())
            .map(CircuitBreakerConfigView::to_core_config)
            .unwrap_or_else(|| default_cb.clone());
        let health = config
            .worker_health
            .get(worker.url())
            .map(HealthConfigView::to_core_config)
            .unwrap_or_else(|| default_health.clone());
        worker.circuit_breaker().update_config(cb);
        worker.update_health_config(health);
    }
}

fn load_config_store(state_file: Option<&PathBuf>) -> Result<Option<ConfigStore>> {
    let Some(path) = state_file else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read state file {}", path.display()))?;
    let store = serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse state file {}", path.display()))?;
    Ok(Some(store))
}

fn initial_persisted_config(
    cli: &Cli,
    router_config: &RouterConfig,
    persisted_store: Option<&ConfigStore>,
) -> Result<PersistedConfig> {
    if let Some(store) = persisted_store {
        return Ok(store.current.clone());
    }

    Ok(PersistedConfig {
        workers: cli
            .worker_urls
            .iter()
            .map(|worker_url| WorkerConfigRequest {
                url: worker_url.clone(),
                model_id: None,
                priority: None,
                cost: None,
                worker_type: Some("regular".to_string()),
                bootstrap_port: None,
                labels: HashMap::new(),
            })
            .collect(),
        default_policy: policy_request_from_config(parse_policy_config(&cli.policy)?),
        model_policies: HashMap::new(),
        default_circuit_breaker: CircuitBreakerConfigView::from_config(
            &default_circuit_breaker_config(router_config),
        ),
        default_health: HealthConfigView::from_config(&default_health_config(router_config)),
        worker_circuit_breakers: HashMap::new(),
        worker_health: HashMap::new(),
    })
}

fn policy_request_from_config(config: PolicyConfig) -> PolicyUpdateRequest {
    match config {
        PolicyConfig::Random => PolicyUpdateRequest {
            policy: "random".to_string(),
            ..PolicyUpdateRequest::default()
        },
        PolicyConfig::RoundRobin => PolicyUpdateRequest {
            policy: "round_robin".to_string(),
            ..PolicyUpdateRequest::default()
        },
        PolicyConfig::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
        } => PolicyUpdateRequest {
            policy: "cache_aware".to_string(),
            cache_threshold: Some(cache_threshold),
            balance_abs_threshold: Some(balance_abs_threshold),
            balance_rel_threshold: Some(balance_rel_threshold),
            eviction_interval_secs: Some(eviction_interval_secs),
            max_tree_size: Some(max_tree_size),
            ..PolicyUpdateRequest::default()
        },
        PolicyConfig::PowerOfTwo {
            load_check_interval_secs,
        } => PolicyUpdateRequest {
            policy: "power_of_two".to_string(),
            load_check_interval_secs: Some(load_check_interval_secs),
            ..PolicyUpdateRequest::default()
        },
        PolicyConfig::ConsistentHash { virtual_nodes } => PolicyUpdateRequest {
            policy: "consistent_hash".to_string(),
            virtual_nodes: Some(virtual_nodes),
            ..PolicyUpdateRequest::default()
        },
        PolicyConfig::RendezvousHash => PolicyUpdateRequest {
            policy: "rendezvous_hash".to_string(),
            ..PolicyUpdateRequest::default()
        },
    }
}

fn persist_revision(state: &ControlState, description: &str) -> Result<()> {
    let snapshot = {
        let mut store = state.config_store.write().unwrap();
        store.push_revision(description.to_string(), state.max_revisions);
        store.clone()
    };

    if let Some(path) = state.state_file.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create state dir {}", parent.display()))?;
        }
        let contents = serde_json::to_string_pretty(&snapshot)?;
        fs::write(path, contents)
            .with_context(|| format!("failed to write state file {}", path.display()))?;
    }
    Ok(())
}

impl Default for PolicyUpdateRequest {
    fn default() -> Self {
        Self {
            policy: "round_robin".to_string(),
            cache_threshold: None,
            balance_abs_threshold: None,
            balance_rel_threshold: None,
            eviction_interval_secs: None,
            max_tree_size: None,
            load_check_interval_secs: None,
            virtual_nodes: None,
        }
    }
}

impl ConfigStore {
    fn new(current: PersistedConfig) -> Self {
        Self {
            current,
            revisions: Vec::new(),
        }
    }

    fn current_revision_id(&self) -> Option<u64> {
        self.revisions.last().map(|revision| revision.id)
    }

    fn push_revision(&mut self, description: String, max_revisions: usize) {
        let id = self.current_revision_id().unwrap_or(0) + 1;
        self.revisions.push(ConfigRevision {
            id,
            timestamp_secs: now_secs(),
            description,
            config: self.current.clone(),
        });

        let keep = max_revisions.max(1);
        if self.revisions.len() > keep {
            let remove_count = self.revisions.len() - keep;
            self.revisions.drain(0..remove_count);
        }
    }
}

impl ConfigRevisionSummary {
    fn from_revision(revision: &ConfigRevision) -> Self {
        Self {
            id: revision.id,
            timestamp_secs: revision.timestamp_secs,
            description: revision.description.clone(),
            workers: revision.config.workers.len(),
            model_policies: revision.config.model_policies.len(),
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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
