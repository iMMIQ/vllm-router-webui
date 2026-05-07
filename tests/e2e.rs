use axum::{
    extract::State,
    http::{HeaderMap, HeaderValue},
    routing::{get, post},
    Json, Router,
};
use portpicker::pick_unused_port;
use serde_json::{json, Value};
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
    }
}

async fn spawn_webui() -> TestServer {
    let port = pick_unused_port().expect("free webui port");
    let app = build_application(test_cli(port))
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
            "labels": {"policy": "round_robin"}
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
