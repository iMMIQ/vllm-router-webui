# vLLM Router WebUI

Single-binary WebUI and control plane for `vllm-project/router`.

The upstream router is imported as a Git submodule under `vendor/router`. This
crate builds one binary that serves:

- the WebUI at `/`
- control APIs under `/api/*`
- upstream router APIs such as `/v1/completions`, `/v1/chat/completions`,
  `/workers`, `/list_workers`, and health endpoints

## Run

```bash
cargo run --bin vllm-router-webui -- --host 127.0.0.1 --port 30000
```

With an initial worker:

```bash
cargo run --bin vllm-router-webui -- \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-url http://127.0.0.1:8000 \
  --policy round_robin \
  --prometheus-host 127.0.0.1 \
  --prometheus-port 29000
```

## Build

```bash
cargo build --bin vllm-router-webui
```

The binary is written to:

```bash
target/debug/vllm-router-webui
```

## Test

```bash
cargo test
```

The end-to-end test suite starts the WebUI/router server and mock workers over
HTTP. It verifies:

- WebUI route availability
- dynamic worker add/list/remove
- worker replacement without removing the old worker when the new worker is not
  healthy
- runtime policy updates
- worker detail monitoring with health, availability, load, processed counters,
  and circuit breaker state
- circuit breaker force-open and reset controls
- optional Prometheus metrics server
- built-in WebUI metrics summary
- proxied `/v1/completions` routing through the upstream router

## Control API

`GET /api/overview`

Returns worker inventory, worker stats, default policy, and model policy
mappings. Each worker includes health, availability, load, processed request
count, metadata, and circuit breaker details.

`POST /api/workers`

Adds a worker after checking its health endpoint.

```json
{
  "url": "http://127.0.0.1:8000",
  "model_id": "test-model",
  "worker_type": "regular",
  "labels": {
    "policy": "round_robin"
  }
}
```

`DELETE /api/workers`

Removes a worker by URL.

```json
{
  "url": "http://127.0.0.1:8000"
}
```

`POST /api/workers/replace`

Adds the replacement worker first and removes the old worker only after the new
worker passed health validation and registration.

`POST /api/workers/circuit/open`

Forces a worker circuit breaker open, which makes the worker unavailable for new
routed requests.

```json
{
  "url": "http://127.0.0.1:8000"
}
```

`POST /api/workers/circuit/reset`

Resets a worker circuit breaker to closed.

```json
{
  "url": "http://127.0.0.1:8000"
}
```

`POST /api/policies/default`

Updates the default policy for new models.

`POST /api/policies/models/{model_id}`

Updates the policy for an existing model without restarting the server.

## Monitoring

The WebUI shows a built-in summary derived from the in-process worker registry:

- total workers
- healthy workers
- unavailable workers
- open circuits
- max worker load
- total processed request counter exposed by workers

Worker rows include:

- health and availability
- circuit breaker state
- load and processed request count
- circuit breaker failure counters
- reset/open/remove controls

Prometheus can be enabled with:

```bash
cargo run --bin vllm-router-webui -- \
  --host 127.0.0.1 \
  --port 30000 \
  --prometheus-host 127.0.0.1 \
  --prometheus-port 29000
```

Metrics are exposed at:

```text
http://127.0.0.1:29000/metrics
```

The upstream router metrics include request counters, request latency
histograms, worker health/load gauges, policy decision counters, retry metrics,
PD metrics, and circuit breaker metrics such as `vllm_router_cb_state`.

## Upstream Changes

The wrapper avoids changing upstream request routing and worker implementations.
The submodule contains a small control-plane-oriented patch that makes the
upstream policy registry mutable at runtime:

- mutable default policy
- set/remove model-specific policy
- cache-aware worker removal cleanup fix in the regular router path
