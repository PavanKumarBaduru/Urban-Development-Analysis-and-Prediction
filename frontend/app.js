// ── config ────────────────────────────────────────────────────────────────────
const API_BASE = "http://127.0.0.1:8000";

// ── state ─────────────────────────────────────────────────────────────────────
let selectedFiles = [];
let charts = {};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const fileInput     = document.getElementById("file-input");
const fileCount     = document.getElementById("file-count");
const analyzeBtn    = document.getElementById("analyze-btn");
const loader        = document.getElementById("loader");
const resultsDiv    = document.getElementById("results");
const errorDiv      = document.getElementById("error-div");

// ── file selection ─────────────────────────────────────────────────────────────
fileInput.addEventListener("change", () => {
  selectedFiles = Array.from(fileInput.files)
    .filter(f => f.name.toLowerCase().endsWith(".tif") || f.name.toLowerCase().endsWith(".tiff"));
  fileCount.textContent = selectedFiles.length > 0
    ? `✓ ${selectedFiles.length} image(s) selected`
    : "No valid .tif files selected";
  analyzeBtn.disabled = selectedFiles.length < 2;
});

// ── tabs ───────────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// ── analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  errorDiv.style.display = "none";
  resultsDiv.style.display = "none";
  loader.style.display = "flex";
  analyzeBtn.disabled = true;

  const formData = new FormData();
  // sort files by name so temporal order is preserved
  const sorted = [...selectedFiles].sort((a, b) => a.name.localeCompare(b.name));
  sorted.forEach(f => formData.append("files", f));

  try {
    const res  = await fetch(`${API_BASE}/analyze`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Server error");
    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    loader.style.display = "none";
    analyzeBtn.disabled = false;
  }
});

// ── render ─────────────────────────────────────────────────────────────────────
function renderResults(data) {
  renderPhase1(data.phase1);
  renderPhase2(data.phase2);
  renderPhase3(data.phase3);
  resultsDiv.style.display = "block";
  // activate first tab
  document.querySelector('.tab-btn[data-tab="tab-p1"]').click();
  resultsDiv.scrollIntoView({ behavior: "smooth" });
}

// ── Phase-1 ────────────────────────────────────────────────────────────────────
function renderPhase1(p1) {
  // stat cards
  document.getElementById("p1-baseline-area").textContent     = p1.baseline_area_acres.toFixed(2);
  document.getElementById("p1-final-area").textContent        = p1.final_area_acres.toFixed(2);
  document.getElementById("p1-total-new-bldg").textContent    = p1.total_new_buildings;
  document.getElementById("p1-area-change").textContent       = p1.total_area_change_acres.toFixed(2);
  document.getElementById("p1-baseline-count").textContent    = p1.baseline_building_count;
  document.getElementById("p1-final-count").textContent       = p1.final_building_count;

  // monthly table
  const tbody = document.getElementById("monthly-tbody");
  tbody.innerHTML = "";
  p1.monthly_records.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.start_date}</td>
      <td>${r.end_date}</td>
      <td>${r.prev_building_count}</td>
      <td>${r.curr_building_count}</td>
      <td style="color:var(--green)">${r.new_building_count}</td>
      <td>${r.prev_area_acres.toFixed(3)}</td>
      <td>${r.curr_area_acres.toFixed(3)}</td>
      <td style="color:var(--accent)">${r.area_change_acres.toFixed(4)}</td>
      <td>${r.season}</td>`;
    tbody.appendChild(tr);
  });

  // Monthly Building Development chart
  destroyChart("chartMonthly");
  const labels = p1.monthly_records.map(r => r.end_date.slice(0, 7));
  const newBldg = p1.monthly_records.map(r => r.new_building_count);
  charts["chartMonthly"] = new Chart(document.getElementById("chartMonthly"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "New Buildings", data: newBldg,
        borderColor: "#38bdf8", backgroundColor: "rgba(56,189,248,0.12)",
        pointRadius: 4, tension: 0.3, fill: true,
      }]
    },
    options: chartOpts("Monthly Building Development", "New Buildings"),
  });

  // Monthly Area Change chart
  destroyChart("chartArea");
  const areaChg = p1.monthly_records.map(r => r.area_change_acres);
  charts["chartArea"] = new Chart(document.getElementById("chartArea"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Area Change (acres)", data: areaChg,
        borderColor: "#818cf8", backgroundColor: "rgba(129,140,248,0.12)",
        pointRadius: 4, tension: 0.3, fill: true,
      }]
    },
    options: chartOpts("Monthly Area Change", "Acres"),
  });

  // Cumulative trend
  destroyChart("chartCumulative");
  let cumBldg = 0, cumArea = 0;
  const cumBldgArr = [], cumAreaArr = [];
  p1.monthly_records.forEach(r => {
    cumBldg += r.new_building_count;
    cumArea += r.area_change_acres;
    cumBldgArr.push(cumBldg);
    cumAreaArr.push(+cumArea.toFixed(3));
  });
  charts["chartCumulative"] = new Chart(document.getElementById("chartCumulative"), {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Cumulative Buildings", data: cumBldgArr, borderColor: "#fb923c", tension: 0.3, fill: false },
        { label: "Cumulative Area (acres)", data: cumAreaArr, borderColor: "#38bdf8", tension: 0.3, fill: false },
      ]
    },
    options: chartOpts("Urban Change Trend (Cumulative)", "Count / Acres"),
  });

  // Seasonal chart
  destroyChart("chartSeasonal");
  const seasons = Object.keys(p1.seasonal_avg);
  const seasonVals = seasons.map(s => p1.seasonal_avg[s]);
  charts["chartSeasonal"] = new Chart(document.getElementById("chartSeasonal"), {
    type: "bar",
    data: {
      labels: seasons,
      datasets: [{
        label: "Avg New Buildings", data: seasonVals,
        backgroundColor: ["#38bdf8","#fb923c","#4ade80","#818cf8"],
      }]
    },
    options: chartOpts("Seasonal Analysis", "Avg Buildings / Month"),
  });
}

// ── Phase-2 ────────────────────────────────────────────────────────────────────
function renderPhase2(p2) {
  document.getElementById("img-last-input").src       = `data:image/png;base64,${p2.last_input_img_b64}`;
  document.getElementById("img-predicted-mask").src   = `data:image/png;base64,${p2.predicted_mask_b64}`;
  document.getElementById("img-highlighted").src      = `data:image/png;base64,${p2.highlighted_img_b64}`;
}

// ── Phase-3 ────────────────────────────────────────────────────────────────────
function renderPhase3(p3) {
  document.getElementById("img-cloud-mask").src    = `data:image/png;base64,${p3.cloud_mask_b64}`;
  document.getElementById("img-cloud-overlay").src = `data:image/png;base64,${p3.cloud_overlay_b64}`;

  // classification result box
  document.getElementById("p3-result-box").innerHTML = `
    <div><span class="key">Baseline State</span></div>
    <div><span class="key">  Baseline Area   </span><span class="sep">:</span> <span class="val">${p3.baseline_area.toFixed(2)} acres</span></div>
    <div style="margin-top:.5rem"><span class="key">Observed Development</span></div>
    <div><span class="key">  Total New Buildings   </span><span class="sep">:</span> <span class="val">${p3.total_new_buildings}</span></div>
    <div><span class="key">  Final Developed Area  </span><span class="sep">:</span> <span class="val">${p3.final_area.toFixed(2)} acres</span></div>
    <div style="margin-top:.5rem"><span class="key">Predicted Development</span></div>
    <div><span class="key">  Area Expected to Grow </span><span class="sep">:</span> <span class="val">${p3.pred_area_acres.toFixed(2)} acres</span></div>
    <div><span class="key">  Buildings Expected    </span><span class="sep">:</span> <span class="val">${p3.pred_buildings}</span></div>
    <div style="margin-top:.5rem"><span class="key">Overall Urban Growth</span></div>
    <div><span class="key">  Growth Percentage     </span><span class="sep">:</span> <span class="val">${p3.growth_percentage}%</span></div>`;

  // badges
  const ubadge = document.getElementById("urban-class-badge");
  const bbadge = document.getElementById("bldg-class-badge");
  ubadge.textContent = p3.urban_class;
  bbadge.textContent = p3.building_class;
  const cls = p3.urban_class.toLowerCase();
  ubadge.className = "badge " + (cls.includes("high") ? "badge-high" : cls.includes("moderate") ? "badge-moderate" : "badge-low");
  bbadge.className = "badge " + (p3.building_class.toLowerCase().includes("high") ? "badge-high" : "badge-low");

  // Urban Development Contribution bar chart
  destroyChart("chartContribution");
  charts["chartContribution"] = new Chart(document.getElementById("chartContribution"), {
    type: "bar",
    data: {
      labels: ["Observed", "Predicted"],
      datasets: [{
        label: "Area (acres)",
        data: [p3.final_area - p3.baseline_area, p3.pred_area_acres],
        backgroundColor: ["#38bdf8", "#818cf8"],
      }]
    },
    options: chartOpts("Urban Development Contribution", "Area (acres)"),
  });
}

// ── chart helpers ──────────────────────────────────────────────────────────────
function chartOpts(title, yLabel) {
  return {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#94a3b8" } },
      title:  { display: true, text: title, color: "#e2e8f0", font: { size: 13 } },
    },
    scales: {
      x: { ticks: { color: "#64748b", maxRotation: 45 }, grid: { color: "#1e293b" } },
      y: { ticks: { color: "#64748b" }, grid: { color: "#1e293b" },
           title: { display: true, text: yLabel, color: "#64748b" } },
    },
  };
}

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function showError(msg) {
  errorDiv.textContent = "⚠ " + msg;
  errorDiv.style.display = "block";
}