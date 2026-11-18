/* PathSnap Field Atlas: evaluator UI. It talks only to the PathSnap API. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
  };

  const state = {
    api: store.get("pathsnap.api", "http://localhost:8000"),
    trips: [], models: [], selTrip: null,
    on: new Set(), mode: {},           // model key -> "hmm" | "clf"
    result: null, layers: {}, idx: 0, playing: false, timer: null,
  };

  // ---------------- map ----------------
  const map = L.map("map", { zoomControl: true, attributionControl: true, preferCanvas: false }).setView([20, 0], 2);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap contributors" }).addTo(map);
  // one pane + renderer per layer so the stacking order is fixed: network < GPS < routes < cursor
  const R = {};
  [["net", 405], ["gps", 408], ["routes", 410], ["cursor", 430]].forEach(([n, z]) => { map.createPane(n); map.getPane(n).style.zIndex = z; R[n] = L.svg({ pane: n, padding: 0.3 }); });
  const svg = R.routes;
  const groups = { net: L.layerGroup().addTo(map), gps: L.layerGroup().addTo(map), routes: L.layerGroup().addTo(map), cursor: L.layerGroup().addTo(map) };
  L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);
  try { new ResizeObserver(() => map.invalidateSize({ pan: false })).observe(document.getElementById("map")); } catch (e) { window.addEventListener("resize", () => map.invalidateSize()); }
  map.on("move", corners);
  function corners() {
    const b = map.getBounds(), f = (v, p, n) => Math.abs(v).toFixed(4) + "°" + (v >= 0 ? p : n);
    $("c-tl").textContent = f(b.getNorth(), "N", "S") + "  " + f(b.getWest(), "E", "W");
    $("c-br").textContent = f(b.getSouth(), "N", "S") + "  " + f(b.getEast(), "E", "W");
  }
  corners();

  // ---------------- wire log ----------------
  function log(html, cls) {
    const d = document.createElement("div"); d.className = "ln " + (cls || "");
    const t = new Date().toTimeString().slice(0, 8);
    d.innerHTML = `<span class="muted">${t}</span>  ${html}`;
    $("log").prepend(d);
  }
  async function call(method, path, body) {
    const t0 = performance.now(); const url = state.api.replace(/\/+$/, "") + path;
    try {
      const r = await fetch(url, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
      const txt = await r.text(); const ms = performance.now() - t0;
      log(`<b>${method}</b> ${path} → <span class="${r.ok ? "ok" : "err"}">${r.status}</span> · ${ms.toFixed(0)} ms · ${(txt.length / 1024).toFixed(1)} kB`, r.ok ? "" : "err");
      if (!r.ok) throw new Error(`${r.status} ${txt.slice(0, 120)}`);
      return JSON.parse(txt);
    } catch (e) {
      if (!(e instanceof Error && /^\d{3}/.test(e.message))) log(`<b>${method}</b> ${path} → <span class="err">unreachable</span> (${e.message})`, "err");
      throw e;
    }
  }
  function status(kind, text) { $("pip").className = "pip " + kind; $("status-text").textContent = text; }

  // ---------------- connect ----------------
  async function connect() {
    return connectImpl();
  }
  async function connectImpl() {
    state.api = $("api").value.trim() || state.api; store.set("pathsnap.api", state.api);
    status("busy", "contacting " + state.api);
    try {
      const h = await call("GET", "/health");
      const [models, trips] = await Promise.all([call("GET", "/models"), call("GET", "/trips")]);
      state.models = models; state.trips = trips;
      const ranked = [...models].sort((a, b) => (b.student_test?.hmm_pointwise || 0) - (a.student_test?.hmm_pointwise || 0));
      state.on = new Set(ranked.slice(0, 3).map((m) => m.key));
      const def = models.find((m) => m.default); if (def) state.on.add(def.key);
      models.forEach((m) => (state.mode[m.key] = state.mode[m.key] || "hmm"));
      status("ok", `connected · ${h.models.length} models · ${h.trips} journeys`);
      renderTrips(); renderModels(); updateRun();
    } catch (e) { status("err", "cannot reach the API: check the IP, port and Wi-Fi"); }
  }

  // ---------------- ledger ----------------
  const fmtDur = (s) => { s = Math.round(s); const m = Math.floor(s / 60); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min ${s % 60} s`; };
  const roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"];
  function renderTrips() {
    const ol = $("trips"); ol.innerHTML = "";
    state.trips.forEach((t, i) => {
      const li = document.createElement("li"); li.className = "trip" + (state.selTrip === t.trip_id ? " sel" : "");
      const lat = (t.bbox[0] + t.bbox[2]) / 2, lon = (t.bbox[1] + t.bbox[3]) / 2;
      li.innerHTML = `<span class="ix">${roman[i] || i + 1}.</span>
        <span><div class="tid">№ ${t.trip_id}</div><div class="tmeta">${fmtDur(t.duration_s)} · ${lat.toFixed(2)}°, ${lon.toFixed(2)}°</div></span>
        <span class="tpts">${t.points.toLocaleString()}<br><span class="tmeta">pts</span></span>`;
      li.onclick = () => { state.selTrip = t.trip_id; renderTrips(); updateRun(); previewTrip(t); };
      ol.appendChild(li);
    });
  }
  function previewTrip(t) {
    $("plate-title").textContent = `Journey № ${t.trip_id}`;
    $("plate-meta").innerHTML = `${t.points.toLocaleString()} observations<br>${fmtDur(t.duration_s)}`;
    map.flyToBounds([[t.bbox[0], t.bbox[1]], [t.bbox[2], t.bbox[3]]], { padding: [30, 30], duration: 0.9 });
  }
  function renderModels() {
    const ul = $("models"); ul.innerHTML = "";
    state.models.forEach((m) => {
      const li = document.createElement("li"); const on = state.on.has(m.key);
      li.className = "model " + (on ? "on" : "off"); li.style.color = m.colour;
      const st = m.student_test || {};
      li.innerHTML = `<span class="sw"></span>
        <span class="mname" style="color:var(--ink)">${m.label}${m.default ? '<span class="badge" style="color:' + m.colour + '">default</span>' : ""}</span>
        <span class="seg" data-k="${m.key}"><button data-m="hmm" class="${state.mode[m.key] === "hmm" ? "on" : ""}">HMM</button><button data-m="clf" class="${state.mode[m.key] === "clf" ? "on" : ""}">CLF</button></span>
        <span class="mstat">dev-test: top-1 ${fmt(st.top1)} · +HMM ${fmt(st.hmm_pointwise)} · F1 ${fmt(st.macro_f1)}</span>`;
      li.onclick = (ev) => {
        const b = ev.target.closest("button");
        if (b) { state.mode[m.key] = b.dataset.m; renderModels(); if (state.result) drawRoutes(false); return; }
        if (state.on.has(m.key)) state.on.delete(m.key); else state.on.add(m.key);
        renderModels(); updateRun();
        if (state.result) drawRoutes(false);
      };
      ul.appendChild(li);
    });
    renderKey();
  }
  function renderKey() {
    $("key-models").innerHTML = state.models.filter((m) => state.on.has(m.key)).map((m) =>
      `<div class="key-row" style="color:${m.colour}"><span class="sw-line ${state.mode[m.key] === "clf" ? "dash" : ""}"></span><span style="color:var(--ink)">${m.label} · ${state.mode[m.key].toUpperCase()}</span></div>`).join("");
  }
  const fmt = (v, d = 3) => (v === null || v === undefined || isNaN(v) ? "–" : Number(v).toFixed(d));
  function updateRun() {
    const ok = state.selTrip && state.on.size > 0;
    $("run").disabled = !ok;
    $("run-sub").textContent = !state.selTrip ? "select a journey" : state.on.size === 0 ? "select at least one model" : `№ ${state.selTrip} · ${state.on.size} model${state.on.size > 1 ? "s" : ""}`;
  }

  // ---------------- run ----------------
  async function run() {
    if ($("run").disabled) return;
    stop();
    $("veil").classList.add("show"); $("veil-text").textContent = `Surveying journey № ${state.selTrip}…`;
    status("busy", "reconstructing…");
    try {
      const keys = state.models.map((m) => m.key);   // ask for every model so toggles are instant
      const res = await call("POST", "/predict", { trip_id: state.selTrip, models: keys });
      state.result = res; state.idx = 0;
      status("ok", `reconstructed № ${res.trip_id} in ${(res.total_ms / 1000).toFixed(2)} s`);
      drawBase(); drawRoutes(true); renderCards(); setupTimeline();
    } catch (e) { status("err", "reconstruction failed: " + e.message); }
    finally { $("veil").classList.remove("show"); }
  }

  function drawBase() {
    const r = state.result; groups.net.clearLayers(); groups.gps.clearLayers();
    L.polyline(r.network.map((s) => [[s[0], s[1]], [s[2], s[3]]]), { renderer: R.net, color: "#6f675a", weight: 1, opacity: 0.45, interactive: false }).addTo(groups.net);
    const step = Math.max(1, Math.ceil(r.gps.length / 2500));
    r.gps.forEach((g, i) => { if (i % step === 0) L.circleMarker([g[0], g[1]], { renderer: R.gps, radius: 1.6, stroke: false, fill: true, fillColor: "#1d1b17", fillOpacity: 0.4, interactive: false }).addTo(groups.gps); });
    const lats = r.gps.map((g) => g[0]), lons = r.gps.map((g) => g[1]);
    map.fitBounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]], { padding: [28, 28] });
    $("plate-title").textContent = `Journey № ${r.trip_id}`;
    const dur = r.gps.length ? r.gps[r.gps.length - 1][2] - r.gps[0][2] : 0;
    $("plate-meta").innerHTML = `${r.points.toLocaleString()} observations · ${r.candidate_rows.toLocaleString()} candidates<br>${fmtDur(dur)} · server ${r.total_ms.toFixed(0)} ms${r.cached ? " (cached)" : ""}`;
  }

  function drawRoutes(animate) {
    const r = state.result; if (!r) return;
    groups.routes.clearLayers(); state.layers = {};
    const active = r.models.filter((m) => state.on.has(m.key));
    active.forEach((m, k) => {
      const mode = state.mode[m.key]; const lines = mode === "hmm" ? m.hmm_route : m.clf_route;
      // transit-map ribbons: each model is drawn as a parallel line offset by a few pixels,
      // so routes run side by side where models agree and split apart where they disagree
      const weight = 3.2, offset = (k - (active.length - 1) / 2) * 3.6;
      const ls = lines.map((pl) => L.polyline(pl, { renderer: svg, color: m.colour, weight, opacity: 0.95, offset, className: "route-path", dashArray: mode === "clf" ? "5 5" : null, lineCap: "butt", lineJoin: "round" }));
      ls.forEach((l) => l.addTo(groups.routes));
      state.layers[m.key] = ls;
      if (animate && mode === "hmm" && new URLSearchParams(location.search).get("anim") !== "0") requestAnimationFrame(() => ls.forEach((l) => {
        const p = l.getElement(); if (!p) return; const len = p.getTotalLength();
        p.style.strokeDasharray = `${len} ${len}`; p.style.strokeDashoffset = len;
        p.style.setProperty("--dur", `${1.4 + 0.25 * k + Math.min(len / 4000, 1.6)}s`);
        p.style.animationDelay = `${0.35 * k}s`; p.classList.add("route-draw");
      }));
    });
    renderKey(); cursor();
  }

  // ---------------- findings ----------------
  function renderCards() {
    const r = state.result; const box = $("cards"); box.innerHTML = "";
    const ms = [...r.models].sort((a, b) => (b.metrics.accuracy_hmm ?? 0) - (a.metrics.accuracy_hmm ?? 0));
    ms.forEach((m, i) => {
      const x = m.metrics; const d = document.createElement("div"); d.className = "card"; d.style.setProperty("--c", m.colour); d.style.animationDelay = `${0.12 * i}s`;
      const bars = [["Classifier", x.accuracy_clf], ["+ HMM", x.accuracy_hmm], ["Route F1", x.route_f1], ["Recall", x.route_recall], ["Continuity", x.continuity_hmm]];
      d.innerHTML = `<div class="card-top"><span class="card-name">${m.label}</span><span class="card-rank">${["first", "second", "third", "fourth", "fifth"][i] || i + 1}</span></div>
        <div class="big">${x.accuracy_hmm == null ? "–" : (100 * x.accuracy_hmm).toFixed(1)}<small>% points matched (HMM)</small></div>
        <div class="bars">${bars.map(([l, v]) => `<span class="lbl">${l}</span><div class="bar"><span data-w="${v == null ? 0 : (v * 100).toFixed(1)}" class="${l === "Classifier" ? "ghost" : ""}"></span></div><span class="val">${fmt(v)}</span>`).join("")}</div>
        <div class="card-foot"><span>${m.n_route_arcs} arcs in route</span><span>${m.infer_ms.toFixed(0)} ms inference</span></div>`;
      box.appendChild(d);
    });
    requestAnimationFrame(() => setTimeout(() => box.querySelectorAll(".bar span").forEach((s) => {
      const w = parseFloat(s.dataset.w); s.style.width = Math.max(0, (w - 50) * 2) + "%";   // axis shows 50 to 100 %
    }), 60));
    const a = ms[0], b = ms[1];
    let v = `On journey № ${r.trip_id}, <b>${a.label}</b> matched ${(100 * (a.metrics.accuracy_hmm || 0)).toFixed(1)}% of GPS points`;
    if (b) v += `, ${(100 * ((a.metrics.accuracy_hmm || 0) - (b.metrics.accuracy_hmm || 0))).toFixed(1)} points ahead of ${b.label}`;
    const gain = (a.metrics.accuracy_hmm || 0) - (a.metrics.accuracy_clf || 0);
    v += `. Viterbi decoding ${gain >= 0 ? "added" : "cost"} ${(100 * Math.abs(gain)).toFixed(1)} points over the classifier alone and kept ${(100 * (a.metrics.continuity_hmm || 0)).toFixed(1)}% of steps connected. (Bars show 50 to 100%. The outlined bar is classifier-only.)`;
    $("verdict").innerHTML = v;
  }

  // ---------------- timeline ----------------
  function setupTimeline() {
    const n = state.result.gps.length; const s = $("scrub");
    s.max = Math.max(0, n - 1); s.value = 0; s.disabled = n === 0; $("play").disabled = n === 0;
    cursor();
  }
  function cursor() {
    const r = state.result; groups.cursor.clearLayers(); if (!r || !r.gps.length) return;
    const i = Math.min(state.idx, r.gps.length - 1); const g = r.gps[i];
    const t0 = r.gps[0][2];
    $("clock").textContent = `t = ${fmtDur(g[2] - t0)}`; $("ptno").textContent = `point ${i + 1} / ${r.gps.length}`;
    const ro = []; const arcs = {};
    r.models.filter((m) => state.on.has(m.key)).forEach((m) => {
      const mp = (state.mode[m.key] === "hmm" ? m.matched_hmm : m.matched_clf)[i];
      if (mp) {
        L.polyline([[g[0], g[1]], [mp[0], mp[1]]], { renderer: R.cursor, color: m.colour, weight: 1, dashArray: "2 3", interactive: false }).addTo(groups.cursor);
        L.marker([mp[0], mp[1]], { pane: "cursor", interactive: false, icon: L.divIcon({ className: "", iconSize: [12, 12], iconAnchor: [6, 6],
          html: `<svg width="12" height="12"><rect x="2" y="2" width="8" height="8" transform="rotate(45 6 6)" fill="${m.colour}" stroke="#f7f2e7" stroke-width="1.2"/></svg>` }) }).addTo(groups.cursor);
        arcs[mp[2]] = (arcs[mp[2]] || 0) + 1;
      }
      ro.push({ m, arc: mp ? mp[2] : null });
    });
    L.circleMarker([g[0], g[1]], { renderer: R.cursor, radius: 7, color: "#1d1b17", weight: 1.5, fill: false }).addTo(groups.cursor);
    L.circleMarker([g[0], g[1]], { renderer: R.cursor, radius: 2.5, color: "#1d1b17", weight: 0, fillOpacity: 1 }).addTo(groups.cursor);
    const agree = Object.keys(arcs).length === 1 && ro.length > 1;
    $("readout").innerHTML = ro.map(({ m, arc }) => `<span class="ro" style="color:${m.colour}"><i></i><span style="color:var(--ink)">${m.label}: arc ${arc ?? "–"}</span></span>`).join("") +
      (ro.length > 1 ? `<span class="ro muted">${agree ? "· all models agree" : "· models disagree here"}</span>` : "");
    if (state.playing && !map.getBounds().pad(-0.15).contains([g[0], g[1]])) map.panTo([g[0], g[1]], { animate: true, duration: 0.4 });
  }
  function play() {
    if (!state.result) return;
    if (state.playing) return stop();
    state.playing = true; $("play").classList.add("on"); $("play").textContent = "❚❚";
    const n = state.result.gps.length; const step = Math.max(1, Math.round(n / 900));
    state.timer = setInterval(() => {
      state.idx += step; if (state.idx >= n) { state.idx = n - 1; stop(); }
      $("scrub").value = state.idx; cursor();
    }, 33);
  }
  function stop() { state.playing = false; clearInterval(state.timer); $("play").classList.remove("on"); $("play").textContent = "▶"; }

  // ---------------- wire up ----------------
  const qs = new URLSearchParams(location.search);
  if (qs.get("api")) state.api = qs.get("api");
  $("api").value = state.api;
  $("connect").onclick = connect;
  $("api").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  $("run").onclick = run;
  $("play").onclick = play;
  $("scrub").addEventListener("input", (e) => { state.idx = +e.target.value; cursor(); });
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
    if (e.code === "Space" && state.result) { e.preventDefault(); play(); }
    if (state.result && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
      state.idx = Math.max(0, Math.min(state.result.gps.length - 1, state.idx + (e.key === "ArrowRight" ? 5 : -5))); $("scrub").value = state.idx; cursor();
    }
  });
  connect().then(() => {
    const t = qs.get("trip");
    if (t && state.trips.some((x) => x.trip_id === t)) {
      state.selTrip = t; renderTrips(); updateRun(); previewTrip(state.trips.find((x) => x.trip_id === t));
      if (qs.get("run") === "1") run().then(() => { const i = parseInt(qs.get("t") || "0", 10); if (state.result && i) { state.idx = Math.min(i, state.result.gps.length - 1); $("scrub").value = state.idx; cursor(); } });
    }
  });
})();
