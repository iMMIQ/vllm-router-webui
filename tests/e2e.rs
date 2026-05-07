use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue},
    routing::{get, post},
    Json, Router,
};
use portpicker::pick_unused_port;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::{net::TcpListener, task::JoinHandle, time::Duration};
use vllm_router_webui::{build_application, Cli};

#[derive(Clone)]
struct MockWorkerState {
    id: String,
    model: String,
}

struct TestServer {
    base_url: String,
    handle: JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

fn test_cli(port: u16) -> Cli {
    Cli {
        host: "127.0.0.1".to_string(),
        port,
        worker_urls: Vec::new(),
        policy: "round_robin".to_string(),
        worker_startup_timeout_secs: 1,
        health_endpoint: "/health".to_string(),
        max_payload_size: 16 * 1024 * 1024,
        request_timeout_secs: 10,
        max_concurrent_requests: 128,
        prometheus_host: "127.0.0.1".to_string(),
        prometheus_port: None,
        state_file: None,
        max_revisions: 50,
    }
}

fn temp_state_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vllm-router-webui-{name}-{}-{}.json",
        std::process::id(),
        pick_unused_port().expect("unique suffix")
    ))
}

async fn spawn_webui() -> TestServer {
    spawn_webui_with_cli(test_cli(pick_unused_port().expect("free webui port"))).await
}

async fn spawn_webui_with_cli(cli: Cli) -> TestServer {
    let port = pick_unused_port().expect("free webui port");
    let mut cli = cli;
    cli.port = port;
    let app = build_application(cli.clone())
        .await
        .expect("build webui app");
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind webui");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve webui");
    });
    let base_url = format!("http://127.0.0.1:{port}");
    wait_for_http_ok(&format!("{base_url}/liveness")).await;
    TestServer { base_url, handle }
}

async fn spawn_worker(id: &str, model: &str) -> TestServer {
    let port = pick_unused_port().expect("free worker port");
    let state = Arc::new(MockWorkerState {
        id: id.to_string(),
        model: model.to_string(),
    });
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/get_server_info", get(get_server_info))
        .route("/v1/completions", post(completion))
        .route("/v1/chat/completions", post(chat_completion))
        .with_state(state);
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind worker");
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve worker");
    });
    let base_url = format!("http://127.0.0.1:{port}");
    wait_for_http_ok(&format!("{base_url}/health")).await;
    TestServer { base_url, handle }
}

async fn wait_for_http_ok(url: &str) {
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        if let Ok(response) = client.get(url).send().await {
            if response.status().is_success() {
                return;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {url}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn get_server_info(State(state): State<Arc<MockWorkerState>>) -> Json<Value> {
    Json(json!({
        "model_id": state.model,
        "priority": 50,
        "cost": 1.0,
        "worker_type": "regular"
    }))
}

async fn completion(
    State(state): State<Arc<MockWorkerState>>,
    Json(_body): Json<Value>,
) -> impl axum::response::IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert("x-worker-id", HeaderValue::from_str(&state.id).unwrap());
    (
        headers,
        Json(json!({
            "id": format!("cmpl-{}", state.id),
            "object": "text_completion",
            "model": state.model,
            "choices": [{"index": 0, "text": state.id, "finish_reason": "stop"}],
            "worker_id": state.id
        })),
    )
}

async fn chat_completion(
    State(state): State<Arc<MockWorkerState>>,
    Json(_body): Json<Value>,
) -> impl axum::response::IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert("x-worker-id", HeaderValue::from_str(&state.id).unwrap());
    (
        headers,
        Json(json!({
            "id": format!("chatcmpl-{}", state.id),
            "object": "chat.completion",
            "model": state.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": state.id},
                "finish_reason": "stop"
            }],
            "worker_id": state.id
        })),
    )
}

#[tokio::test]
async fn webui_adds_lists_routes_and_removes_worker() {
    let webui = spawn_webui().await;
    let worker = spawn_worker("worker-a", "test-model").await;
    let client = reqwest::Client::new();

    let html = client
        .get(format!("{}/", webui.base_url))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("vLLM Router WebUI"));

    let add = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker.base_url,
            "model_id": "test-model",
            "worker_type": "regular",
            "labels": {"policy": "power_of_two"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        add.status(),
        reqwest::StatusCode::OK,
        "{}",
        add.text().await.unwrap()
    );

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(overview["stats"]["total_workers"], 1);
    assert_eq!(overview["workers"][0]["model_id"], "test-model");
    assert_eq!(overview["workers"][0]["circuit_breaker"]["state"], "closed");
    assert_eq!(overview["workers"][0]["is_available"], true);
    assert_eq!(overview["metrics"]["unavailable_workers"], 0);

    let routed = client
        .post(format!("{}/v1/completions", webui.base_url))
        .json(&json!({
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(routed.status(), reqwest::StatusCode::OK);
    assert_eq!(routed.headers().get("x-worker-id").unwrap(), "worker-a");
    let routed_body: Value = routed.json().await.unwrap();
    assert_eq!(routed_body["worker_id"], "worker-a");

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(overview["metrics"]["total_processed_requests"].is_number());

    let removed = client
        .delete(format!("{}/api/workers", webui.base_url))
        .json(&json!({ "url": worker.base_url }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        removed.status(),
        reqwest::StatusCode::OK,
        "{}",
        removed.text().await.unwrap()
    );

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(overview["stats"]["total_workers"], 0);
}

#[tokio::test]
async fn replacement_keeps_old_worker_when_new_worker_is_not_ready() {
    let webui = spawn_webui().await;
    let worker = spawn_worker("worker-a", "test-model").await;
    let unused_port = pick_unused_port().expect("free unused port");
    let client = reqwest::Client::new();

    let add = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add.status(), reqwest::StatusCode::OK);

    let replace = client
        .post(format!("{}/api/workers/replace", webui.base_url))
        .json(&json!({
            "old_url": worker.base_url,
            "new_worker": {
                "url": format!("http://127.0.0.1:{unused_port}"),
                "model_id": "test-model",
                "worker_type": "regular"
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(replace.status(), reqwest::StatusCode::BAD_REQUEST);

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(overview["stats"]["total_workers"], 1);

    let routed = client
        .post(format!("{}/v1/completions", webui.base_url))
        .json(&json!({
            "model": "test-model",
            "prompt": "still alive",
            "max_tokens": 1,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(routed.status(), reqwest::StatusCode::OK);
    assert_eq!(routed.headers().get("x-worker-id").unwrap(), "worker-a");
}

#[tokio::test]
async fn replaces_worker_and_updates_policies_without_restarting_server() {
    let webui = spawn_webui().await;
    let worker_a = spawn_worker("worker-a", "test-model").await;
    let worker_b = spawn_worker("worker-b", "test-model").await;
    let client = reqwest::Client::new();

    let add = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker_a.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add.status(), reqwest::StatusCode::OK);

    let policy = client
        .post(format!("{}/api/policies/models/test-model", webui.base_url))
        .json(&json!({ "policy": "random" }))
        .send()
        .await
        .unwrap();
    assert_eq!(policy.status(), reqwest::StatusCode::OK);

    let replace = client
        .post(format!("{}/api/workers/replace", webui.base_url))
        .json(&json!({
            "old_url": worker_a.base_url,
            "new_worker": {
                "url": worker_b.base_url,
                "model_id": "test-model",
                "worker_type": "regular"
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        replace.status(),
        reqwest::StatusCode::OK,
        "{}",
        replace.text().await.unwrap()
    );

    let default_policy = client
        .post(format!("{}/api/policies/default", webui.base_url))
        .json(&json!({ "policy": "rendezvous_hash" }))
        .send()
        .await
        .unwrap();
    assert_eq!(default_policy.status(), reqwest::StatusCode::OK);

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(overview["stats"]["total_workers"], 1);
    assert_eq!(overview["workers"][0]["url"], worker_b.base_url);
    assert_eq!(overview["model_policies"]["test-model"], "random");
    assert_eq!(overview["default_policy"], "rendezvous_hash");

    let routed = client
        .post(format!("{}/v1/completions", webui.base_url))
        .json(&json!({
            "model": "test-model",
            "prompt": "after replace",
            "max_tokens": 1,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(routed.status(), reqwest::StatusCode::OK);
    assert_eq!(routed.headers().get("x-worker-id").unwrap(), "worker-b");
}

#[tokio::test]
async fn circuit_control_opens_and_resets_worker_availability() {
    let webui = spawn_webui().await;
    let worker = spawn_worker("worker-a", "test-model").await;
    let client = reqwest::Client::new();

    let add = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add.status(), reqwest::StatusCode::OK);

    let opened = client
        .post(format!("{}/api/workers/circuit/open", webui.base_url))
        .json(&json!({ "url": worker.base_url }))
        .send()
        .await
        .unwrap();
    assert_eq!(opened.status(), reqwest::StatusCode::OK);
    let opened_body: Value = opened.json().await.unwrap();
    assert_eq!(opened_body["worker"]["circuit_breaker"]["state"], "open");
    assert_eq!(opened_body["worker"]["is_available"], false);

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(overview["metrics"]["open_circuits"], 1);
    assert_eq!(overview["metrics"]["unavailable_workers"], 1);

    let rejected = client
        .post(format!("{}/v1/completions", webui.base_url))
        .json(&json!({
            "model": "test-model",
            "prompt": "blocked",
            "max_tokens": 1,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert!(!rejected.status().is_success());

    let reset = client
        .post(format!("{}/api/workers/circuit/reset", webui.base_url))
        .json(&json!({ "url": worker.base_url }))
        .send()
        .await
        .unwrap();
    assert_eq!(reset.status(), reqwest::StatusCode::OK);
    let reset_body: Value = reset.json().await.unwrap();
    assert_eq!(reset_body["worker"]["circuit_breaker"]["state"], "closed");
    assert_eq!(reset_body["worker"]["is_available"], true);

    let routed = client
        .post(format!("{}/v1/completions", webui.base_url))
        .json(&json!({
            "model": "test-model",
            "prompt": "after reset",
            "max_tokens": 1,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(routed.status(), reqwest::StatusCode::OK);
    assert_eq!(routed.headers().get("x-worker-id").unwrap(), "worker-a");
}

#[tokio::test]
async fn runtime_configs_update_without_interrupting_routing() {
    let webui = spawn_webui().await;
    let worker = spawn_worker("worker-a", "test-model").await;
    let client = reqwest::Client::new();

    let add = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add.status(), reqwest::StatusCode::OK);

    let circuit = client
        .post(format!("{}/api/runtime/circuit-breaker", webui.base_url))
        .json(&json!({
            "failure_threshold": 2,
            "success_threshold": 1,
            "timeout_duration_secs": 7,
            "window_duration_secs": 11
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        circuit.status(),
        reqwest::StatusCode::OK,
        "{}",
        circuit.text().await.unwrap()
    );

    let health = client
        .post(format!("{}/api/runtime/health", webui.base_url))
        .json(&json!({
            "timeout_secs": 2,
            "check_interval_secs": 3,
            "endpoint": "/health",
            "failure_threshold": 4,
            "success_threshold": 1
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        health.status(),
        reqwest::StatusCode::OK,
        "{}",
        health.text().await.unwrap()
    );

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        overview["runtime_config"]["circuit_breaker"]["failure_threshold"],
        2
    );
    assert_eq!(
        overview["workers"][0]["circuit_breaker"]["config"]["timeout_duration_secs"],
        7
    );
    assert_eq!(
        overview["runtime_config"]["health_check"]["timeout_secs"],
        2
    );
    assert_eq!(
        overview["workers"][0]["health_check"]["failure_threshold"],
        4
    );

    let routed = client
        .post(format!("{}/v1/completions", webui.base_url))
        .json(&json!({
            "model": "test-model",
            "prompt": "still routes",
            "max_tokens": 1,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(routed.status(), reqwest::StatusCode::OK);
    assert_eq!(routed.headers().get("x-worker-id").unwrap(), "worker-a");
}

#[tokio::test]
async fn added_workers_inherit_updated_runtime_defaults() {
    let webui = spawn_webui().await;
    let worker = spawn_worker("worker-a", "test-model").await;
    let client = reqwest::Client::new();

    let circuit = client
        .post(format!("{}/api/runtime/circuit-breaker", webui.base_url))
        .json(&json!({
            "failure_threshold": 3,
            "success_threshold": 1,
            "timeout_duration_secs": 9,
            "window_duration_secs": 13
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(circuit.status(), reqwest::StatusCode::OK);

    let health = client
        .post(format!("{}/api/runtime/health", webui.base_url))
        .json(&json!({
            "timeout_secs": 4,
            "check_interval_secs": 5,
            "endpoint": "/health",
            "failure_threshold": 6,
            "success_threshold": 2
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), reqwest::StatusCode::OK);

    let add = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        add.status(),
        reqwest::StatusCode::OK,
        "{}",
        add.text().await.unwrap()
    );

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        overview["workers"][0]["circuit_breaker"]["config"]["failure_threshold"],
        3
    );
    assert_eq!(
        overview["workers"][0]["circuit_breaker"]["config"]["window_duration_secs"],
        13
    );
    assert_eq!(overview["workers"][0]["health_check"]["timeout_secs"], 4);
    assert_eq!(
        overview["workers"][0]["health_check"]["failure_threshold"],
        6
    );
}

#[tokio::test]
async fn persists_runtime_config_and_restores_after_restart() {
    let state_file = temp_state_file("restore");
    let worker = spawn_worker("worker-a", "test-model").await;
    let client = reqwest::Client::new();

    {
        let mut cli = test_cli(pick_unused_port().expect("free webui port"));
        cli.state_file = Some(state_file.clone());
        let webui = spawn_webui_with_cli(cli).await;

        let add = client
            .post(format!("{}/api/workers", webui.base_url))
            .json(&json!({
                "url": worker.base_url,
                "model_id": "test-model",
                "worker_type": "regular"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(add.status(), reqwest::StatusCode::OK);

        let policy = client
            .post(format!("{}/api/policies/models/test-model", webui.base_url))
            .json(&json!({ "policy": "random" }))
            .send()
            .await
            .unwrap();
        assert_eq!(policy.status(), reqwest::StatusCode::OK);

        let circuit = client
            .post(format!("{}/api/runtime/circuit-breaker", webui.base_url))
            .json(&json!({
                "failure_threshold": 2,
                "success_threshold": 1,
                "timeout_duration_secs": 8,
                "window_duration_secs": 12
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(circuit.status(), reqwest::StatusCode::OK);

        let overview: Value = client
            .get(format!("{}/api/overview", webui.base_url))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(overview["config"]["persistence_enabled"], true);
        assert!(overview["config"]["revision_count"].as_u64().unwrap() >= 4);
    }

    {
        let mut cli = test_cli(pick_unused_port().expect("free webui port"));
        cli.state_file = Some(state_file.clone());
        let webui = spawn_webui_with_cli(cli).await;

        let overview: Value = client
            .get(format!("{}/api/overview", webui.base_url))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(overview["stats"]["total_workers"], 1);
        assert_eq!(overview["workers"][0]["url"], worker.base_url);
        assert_eq!(overview["model_policies"]["test-model"], "random");
        assert_eq!(
            overview["runtime_config"]["circuit_breaker"]["failure_threshold"],
            2
        );
        assert_eq!(
            overview["workers"][0]["circuit_breaker"]["config"]["timeout_duration_secs"],
            8
        );
    }

    let _ = std::fs::remove_file(state_file);
}

#[tokio::test]
async fn config_history_rolls_back_worker_policy_and_runtime_config() {
    let state_file = temp_state_file("rollback");
    let worker_a = spawn_worker("worker-a", "test-model").await;
    let worker_b = spawn_worker("worker-b", "test-model").await;
    let client = reqwest::Client::new();
    let mut cli = test_cli(pick_unused_port().expect("free webui port"));
    cli.state_file = Some(state_file.clone());
    let webui = spawn_webui_with_cli(cli).await;

    let add_a = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker_a.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add_a.status(), reqwest::StatusCode::OK);

    let history: Value = client
        .get(format!("{}/api/config/history", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let rollback_revision = history["current_revision"].as_u64().unwrap();

    let add_b = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker_b.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add_b.status(), reqwest::StatusCode::OK);

    let policy = client
        .post(format!("{}/api/policies/models/test-model", webui.base_url))
        .json(&json!({ "policy": "random" }))
        .send()
        .await
        .unwrap();
    assert_eq!(policy.status(), reqwest::StatusCode::OK);

    let circuit = client
        .post(format!("{}/api/runtime/circuit-breaker", webui.base_url))
        .json(&json!({
            "failure_threshold": 2,
            "timeout_duration_secs": 8
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(circuit.status(), reqwest::StatusCode::OK);

    let rollback = client
        .post(format!("{}/api/config/rollback", webui.base_url))
        .json(&json!({ "revision_id": rollback_revision }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        rollback.status(),
        reqwest::StatusCode::OK,
        "{}",
        rollback.text().await.unwrap()
    );

    let overview: Value = client
        .get(format!("{}/api/overview", webui.base_url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(overview["stats"]["total_workers"], 1);
    assert_eq!(overview["workers"][0]["url"], worker_a.base_url);
    assert!(overview["model_policies"].get("test-model").is_none());
    assert_ne!(
        overview["runtime_config"]["circuit_breaker"]["failure_threshold"],
        2
    );

    let _ = std::fs::remove_file(state_file);
}

#[tokio::test]
async fn prometheus_metrics_server_can_be_enabled() {
    let metrics_port = pick_unused_port().expect("free prometheus port");
    let mut cli = test_cli(pick_unused_port().expect("free webui port"));
    cli.prometheus_port = Some(metrics_port);
    let webui = spawn_webui_with_cli(cli).await;
    let worker = spawn_worker("worker-a", "test-model").await;
    let client = reqwest::Client::new();
    let metrics_url = format!("http://127.0.0.1:{metrics_port}/metrics");

    let add = client
        .post(format!("{}/api/workers", webui.base_url))
        .json(&json!({
            "url": worker.base_url,
            "model_id": "test-model",
            "worker_type": "regular"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add.status(), reqwest::StatusCode::OK);

    let routed = client
        .post(format!("{}/v1/completions", webui.base_url))
        .json(&json!({
            "model": "test-model",
            "prompt": "metrics",
            "max_tokens": 1,
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(routed.status(), reqwest::StatusCode::OK);

    wait_for_http_ok(&metrics_url).await;
    let body = client
        .get(metrics_url)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("vllm_router_requests_total"));
    assert!(body.contains("vllm_router_cb_state"));
}
