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

function render(data) {
  $("total-workers").textContent = data.stats.total_workers;
  $("healthy-workers").textContent = data.stats.healthy_workers;
  $("total-models").textContent = data.stats.total_models;
  $("default-policy").textContent = data.default_policy;
  $("total-processed").textContent = data.metrics.total_processed_requests;
  $("unavailable-workers").textContent = data.metrics.unavailable_workers;
  $("open-circuits").textContent = data.metrics.open_circuits;
  $("max-load").textContent = data.metrics.max_worker_load;

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
      <td>${worker.load}</td>
      <td>${worker.processed_requests}</td>
      <td>${worker.circuit_breaker.consecutive_failures}/${worker.circuit_breaker.total_failures}</td>
      <td>
        <button data-action="reset" data-url="${worker.url}">Reset</button>
        <button data-action="open" data-url="${worker.url}">Open</button>
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
    const payload = { policy: $("policy-name").value };
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

refresh().catch((err) => setStatus(err.message));
