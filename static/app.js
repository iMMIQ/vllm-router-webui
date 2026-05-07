const $ = (id) => document.getElementById(id);

async function request(url, options = {}) {
  const res = await fetch(url, {
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

function setStatus(text, isError = true) {
  const el = $("status");
  el.textContent = text || "";
  el.style.color = isError ? "#bf2e2e" : "#16794c";
}

function optionalNumber(id) {
  const value = $(id).value.trim();
  return value === "" ? undefined : Number(value);
}

function optionalText(id) {
  const value = $(id).value.trim();
  return value === "" ? undefined : value;
}

function cleanPayload(payload) {
  return Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined && value !== "")
  );
}

function setInputValue(id, value) {
  $(id).value = value ?? "";
}

function render(data) {
  $("total-workers").textContent = data.stats.total_workers;
  $("healthy-workers").textContent = data.stats.healthy_workers;
  $("total-models").textContent = data.stats.total_models;
  $("default-policy").textContent = data.default_policy;
  $("total-processed").textContent = data.metrics.total_processed_requests;
  $("unavailable-workers").textContent = data.metrics.unavailable_workers;
  $("open-circuits").textContent = data.metrics.open_circuits;
  $("max-load").textContent = data.metrics.max_worker_load;
  setInputValue("cb-failure-threshold", data.runtime_config.circuit_breaker.failure_threshold);
  setInputValue("cb-success-threshold", data.runtime_config.circuit_breaker.success_threshold);
  setInputValue("cb-timeout-duration-secs", data.runtime_config.circuit_breaker.timeout_duration_secs);
  setInputValue("cb-window-duration-secs", data.runtime_config.circuit_breaker.window_duration_secs);
  setInputValue("health-timeout-secs", data.runtime_config.health_check.timeout_secs);
  setInputValue("health-check-interval-secs", data.runtime_config.health_check.check_interval_secs);
  setInputValue("health-endpoint", data.runtime_config.health_check.endpoint);
  setInputValue("health-failure-threshold", data.runtime_config.health_check.failure_threshold);
  setInputValue("health-success-threshold", data.runtime_config.health_check.success_threshold);

  const tbody = $("workers");
  tbody.innerHTML = "";
  for (const worker of data.workers) {
    const availability = worker.is_available ? "available" : "unavailable";
    const circuit = worker.circuit_breaker.state;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${worker.url}</td>
      <td>${worker.model_id}</td>
      <td>${worker.worker_type}</td>
      <td class="${worker.is_healthy ? "healthy" : "unhealthy"}">${worker.is_healthy ? "healthy" : "unhealthy"}</td>
      <td class="${availability}">${availability}</td>
      <td class="${circuit}">${circuit}</td>
      <td>${worker.circuit_breaker.config.failure_threshold}/${worker.circuit_breaker.config.timeout_duration_secs}s</td>
      <td>${worker.health_check.failure_threshold}/${worker.health_check.timeout_secs}s ${worker.health_check.endpoint}</td>
      <td>${worker.load}</td>
      <td>${worker.processed_requests}</td>
      <td>${worker.circuit_breaker.consecutive_failures}/${worker.circuit_breaker.total_failures}</td>
      <td>
        <button data-action="reset" data-url="${worker.url}">Reset</button>
        <button data-action="open" data-url="${worker.url}">Open</button>
        <button data-action="edit-config" data-url="${worker.url}">Config</button>
        <button data-action="remove" data-url="${worker.url}" class="danger">Remove</button>
      </td>
    `;
    tbody.appendChild(tr);
  }

  tbody.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        const body = JSON.stringify({ url: btn.dataset.url });
        if (btn.dataset.action === "remove") {
          await request("/api/workers", { method: "DELETE", body });
          setStatus("Worker removed", false);
        } else if (btn.dataset.action === "open") {
          await request("/api/workers/circuit/open", { method: "POST", body });
          setStatus("Circuit opened", false);
        } else if (btn.dataset.action === "edit-config") {
          $("circuit-url").value = btn.dataset.url;
          $("health-url").value = btn.dataset.url;
          setStatus("Worker config target selected", false);
        } else {
          await request("/api/workers/circuit/reset", { method: "POST", body });
          setStatus("Circuit reset", false);
        }
        await refresh();
      } catch (err) {
        setStatus(err.message);
      }
    });
  });
}

async function refresh() {
  const data = await request("/api/overview");
  render(data);
}

$("refresh").addEventListener("click", () => refresh().catch((err) => setStatus(err.message)));

$("add-worker").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const payload = {
      url: $("worker-url").value.trim(),
      model_id: $("model-id").value.trim() || null,
      worker_type: $("worker-type").value,
      labels: $("worker-policy").value ? { policy: $("worker-policy").value } : {},
    };
    await request("/api/workers", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    $("add-worker").reset();
    await refresh();
    setStatus("Worker added", false);
  } catch (err) {
    setStatus(err.message);
  }
});

$("set-policy").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const model = $("policy-model").value.trim();
    const payload = cleanPayload({
      policy: $("policy-name").value,
      cache_threshold: optionalNumber("cache-threshold"),
      balance_abs_threshold: optionalNumber("balance-abs-threshold"),
      balance_rel_threshold: optionalNumber("balance-rel-threshold"),
      eviction_interval_secs: optionalNumber("eviction-interval-secs"),
      max_tree_size: optionalNumber("max-tree-size"),
      load_check_interval_secs: optionalNumber("load-check-interval-secs"),
      virtual_nodes: optionalNumber("virtual-nodes"),
    });
    if (model) {
      await request(`/api/policies/models/${encodeURIComponent(model)}`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
    } else {
      await request("/api/policies/default", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    }
    await refresh();
    setStatus("Policy updated", false);
  } catch (err) {
    setStatus(err.message);
  }
});

$("set-circuit").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const payload = cleanPayload({
      url: optionalText("circuit-url"),
      failure_threshold: optionalNumber("cb-failure-threshold"),
      success_threshold: optionalNumber("cb-success-threshold"),
      timeout_duration_secs: optionalNumber("cb-timeout-duration-secs"),
      window_duration_secs: optionalNumber("cb-window-duration-secs"),
    });
    await request("/api/runtime/circuit-breaker", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await refresh();
    setStatus("Circuit config updated", false);
  } catch (err) {
    setStatus(err.message);
  }
});

$("set-health").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    const payload = cleanPayload({
      url: optionalText("health-url"),
      timeout_secs: optionalNumber("health-timeout-secs"),
      check_interval_secs: optionalNumber("health-check-interval-secs"),
      endpoint: optionalText("health-endpoint"),
      failure_threshold: optionalNumber("health-failure-threshold"),
      success_threshold: optionalNumber("health-success-threshold"),
    });
    await request("/api/runtime/health", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await refresh();
    setStatus("Health config updated", false);
  } catch (err) {
    setStatus(err.message);
  }
});

refresh().catch((err) => setStatus(err.message));
