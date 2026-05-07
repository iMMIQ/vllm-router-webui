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
  --policy round_robin
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
- proxied `/v1/completions` routing through the upstream router

## Control API

`GET /api/overview`

Returns worker inventory, worker stats, default policy, and model policy
mappings.

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

`POST /api/policies/default`

Updates the default policy for new models.

`POST /api/policies/models/{model_id}`

Updates the policy for an existing model without restarting the server.

## Upstream Changes

The wrapper avoids changing upstream request routing and worker implementations.
The submodule contains a small control-plane-oriented patch that makes the
upstream policy registry mutable at runtime:

- mutable default policy
- set/remove model-specific policy
- cache-aware worker removal cleanup fix in the regular router path
