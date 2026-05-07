# vLLM Router WebUI

[English](README.md)

`vLLM Router WebUI` 是对 `vllm-project/router` 的单二进制 WebUI 和控制平面封装。

上游 router 以 Git submodule 的形式放在 `vendor/router`。本项目构建出的一个二进制同时提供：

- `/` WebUI
- `/api/*` 控制 API
- 上游 router API，例如 `/v1/completions`、`/v1/chat/completions`、`/workers`、`/list_workers` 和健康检查端点

## 运行

```bash
cargo run --bin vllm-router-webui -- --host 127.0.0.1 --port 30000
```

带初始 worker、状态文件和 Prometheus：

```bash
cargo run --bin vllm-router-webui -- \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-url http://127.0.0.1:8000 \
  --policy round_robin \
  --state-file ./router-state.json \
  --prometheus-host 127.0.0.1 \
  --prometheus-port 29000
```

`--state-file` 用于启用控制平面状态持久化。运行时 worker 变更、策略变更、默认熔断配置、默认健康检查配置、单 worker 运行时配置和配置 revision 都会保存为 JSON。进程重启后，WebUI 会加载该状态，并在重新加入 worker 前执行健康检查。

## 构建

```bash
cargo build --bin vllm-router-webui
```

二进制输出路径：

```bash
target/debug/vllm-router-webui
```

## 测试

```bash
cargo test
```

端到端测试会启动 WebUI/router 服务和 mock worker，并验证：

- WebUI 路由可访问
- 动态添加、查看、删除 worker
- 替换 worker 时，新 worker 未就绪不会移除旧 worker
- 运行时策略更新
- 运行时熔断和健康检查配置更新
- 重启后从持久化配置恢复
- 版本化配置回滚
- worker 健康状态、可用性、负载、处理计数和熔断状态监控
- 强制打开和重置熔断
- 可选 Prometheus 指标服务
- WebUI 内置指标汇总
- 通过上游 router 代理 `/v1/completions`

## 控制 API

`GET /api/overview`

返回 worker 清单、worker 统计、默认策略和 model 策略映射。每个 worker 包含健康状态、可用性、负载、已处理请求数、metadata 和熔断详情。

响应还包含：

- `runtime_config`：默认熔断和健康检查配置
- `config`：持久化状态和当前 revision 信息
- `trends`：WebUI 趋势图使用的近期内存指标采样

`POST /api/workers`

添加 worker。添加前会检查 worker 健康端点。

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

按 URL 删除 worker。

```json
{
  "url": "http://127.0.0.1:8000"
}
```

`POST /api/workers/replace`

先添加替换 worker，只有在新 worker 通过健康检查并注册成功后，才移除旧 worker。

`POST /api/workers/circuit/open`

强制打开某个 worker 的熔断器，使该 worker 不再接收新的路由请求。

```json
{
  "url": "http://127.0.0.1:8000"
}
```

`POST /api/workers/circuit/reset`

将某个 worker 的熔断器重置为 closed。

```json
{
  "url": "http://127.0.0.1:8000"
}
```

`POST /api/policies/default`

更新新 model 使用的默认策略。

`POST /api/policies/models/{model_id}`

在不重启服务的情况下更新某个已有 model 的策略。

策略更新请求支持可选参数：

```json
{
  "policy": "cache_aware",
  "cache_threshold": 0.3,
  "balance_abs_threshold": 64,
  "balance_rel_threshold": 1.5,
  "eviction_interval_secs": 120,
  "max_tree_size": 67108864
}
```

`POST /api/runtime/circuit-breaker`

更新当前和未来 worker 使用的默认熔断配置。传入 `url` 时只更新指定 worker。

```json
{
  "failure_threshold": 5,
  "success_threshold": 2,
  "timeout_duration_secs": 30,
  "window_duration_secs": 60
}
```

`POST /api/runtime/health`

更新当前和未来 worker 使用的默认健康检查配置。传入 `url` 时只更新指定 worker。

```json
{
  "timeout_secs": 5,
  "check_interval_secs": 30,
  "endpoint": "/health",
  "failure_threshold": 3,
  "success_threshold": 2
}
```

`GET /api/config/history`

返回已保存的配置 revision，包括 revision ID、时间戳、描述、worker 数量和 model 策略数量。

`POST /api/config/rollback`

将当前运行态回滚到指定 revision。回滚会应用 worker 集合、默认策略、model 策略、默认运行时配置和单 worker 运行时配置。

```json
{
  "revision_id": 3
}
```

## 监控

WebUI 内置从进程内 worker registry 派生的状态汇总：

- worker 总数
- 健康 worker 数
- 不可用 worker 数
- open circuit 数
- 最大 worker 负载
- worker 暴露的总处理请求计数

Worker 行包含：

- 健康状态和可用性
- 熔断状态
- 单 worker 熔断和健康检查配置
- 负载和已处理请求数
- 熔断失败计数
- reset、open、remove 控制按钮

WebUI 还包含：

- 自动刷新开关
- processed requests 和 open circuits 趋势条
- 持久化状态和当前配置 revision
- 配置历史和 rollback 操作

启用 Prometheus：

```bash
cargo run --bin vllm-router-webui -- \
  --host 127.0.0.1 \
  --port 30000 \
  --prometheus-host 127.0.0.1 \
  --prometheus-port 29000
```

指标地址：

```text
http://127.0.0.1:29000/metrics
```

上游 router 指标包括请求计数、请求延迟直方图、worker health/load gauge、策略决策计数、重试指标、PD 指标，以及 `vllm_router_cb_state` 等熔断指标。

## 上游改动

Wrapper 不改变上游请求路由主体逻辑。`vendor/router` 子模块包含少量面向控制平面的补丁，使部分上游状态可在运行时更新：

- 可变默认策略
- 设置和移除 model 级策略
- regular router 路径中的 cache-aware worker 删除清理修复
- 不重置 worker 状态的运行时熔断配置更新
- 现有 worker 的运行时健康检查配置更新
- consistent hashing 的 virtual node 数量可配置

## License

本项目使用 GPLv3 许可证。详见 [LICENSE](LICENSE)。
