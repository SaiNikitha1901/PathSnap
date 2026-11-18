/* PathSnap evaluator UI — talks only to the FastAPI server; never sees reference routes. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PALETTE = { ensemble: "#e4572e", xgboost: "#2e86ab", lightgbm: "#7b2d8e", random_forest: "#2a9d8f", mlp: "#e9a23b" };
  const EXTRA = ["#3d5a80", "#c44569", "#6a994e"];
  const qs = new URLSearchParams(location.search);
  const ANIM = qs.get("anim") !== "0";

  const state = { api: "", models: [], trips: [], sel: null, on: {}, mode: {}, result: null, idx: 0, playing: null, log: [] };

  // ---------- map ----------
  const map = L.map("map", { zoomControl: false, worldCopyJump: true, zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 110, inertiaDeceleration: 2200, zoomAnimation: true, fadeAnimation: true, markerZoomAnimation: true }).setView([30, 20], 3);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/";
  const BASES = {
    Streets: L.tileLayer(ESRI + "World_Street_Map/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors" }),
    Satellite: L.layerGroup([
      L.tileLayer(ESRI + "World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics" }),
      L.tileLayer(ESRI + "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19 }),
      L.tileLayer(ESRI + "Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, opacity: 0.6 }),
    ]),
    Topographic: L.tileLayer(ESRI + "World_Topo_Map/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors" }),
    OpenStreetMap: L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }),
  };
  BASES.Streets.addTo(map);
  L.control.layers(BASES, null, { position: "bottomright", collapsed: true }).addTo(map);
  [["net", 402], ["gps", 405], ["routes", 410], ["cursor", 440]].forEach(([n, z]) => { map.createPane(n); map.getPane(n).style.zIndex = z; });
  const R = { net: L.canvas({ pane: "net" }), gps: L.canvas({ pane: "gps" }), routes: L.svg({ pane: "routes", padding: 0.5 }), cursor: L.svg({ pane: "cursor" }) };
  const G = { net: L.layerGroup().addTo(map), gps: L.layerGroup().addTo(map), routes: L.layerGroup().addTo(map), cursor: L.layerGroup().addTo(map) };

  // ---------- helpers ----------
  const colour = (m, i) => PALETTE[m.key] || EXTRA[i % EXTRA.length] || m.colour;
  const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1));
  const fmtDur = (s) => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60; return (h ? h + ":" + String(m).padStart(2, "0") : String(m).padStart(2, "0")) + ":" + String(x).padStart(2, "0"); };
  const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch {} } };
  const toast = (msg) => { const t = $("toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 3200); };

  async function call(method, path, body) {
    const t0 = performance.now(); let status = 0;
    try {
      const r = await fetch(state.api + path, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
      status = r.status;
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      return await r.json();
    } finally {
      state.log.unshift({ t: new Date().toLocaleTimeString(), method, path: path + (body && body.trip_id ? ` · ${body.trip_id}` : ""), status, ms: Math.round(performance.now() - t0) });
      renderLog();
    }
  }

  function renderLog() {
    $("logCount").textContent = state.log.length;
    $("logList").innerHTML = state.log.slice(0, 60).map((e) =>
      `<div class="ent"><span class="muted">${e.t}</span><span>${e.method}</span><span>${e.path}</span><span class="${e.status >= 200 && e.status < 300 ? "ok" : "bad"}">${e.status || "ERR"} · ${e.ms}ms</span></div>`).join("");
  }

  // ---------- connect ----------
  async function connect() {
    state.api = $("apiUrl").value.trim().replace(/\/+$/, "");
    if (!/^https?:\/\//.test(state.api)) state.api = "http://" + state.api;
    $("apiUrl").value = state.api; store.set("pathsnap.api", state.api);
    $("connDot").className = "dot off";
    try {
      await call("GET", "/health");
      const [models, trips] = await Promise.all([call("GET", "/models"), call("GET", "/trips")]);
      state.models = models; state.trips = trips;
      models.forEach((m, i) => { m.c = colour(m, i); m.label = m.label.replace("Ensemble (LGBM-tuned + XGB)", "Ensemble").replace("LightGBM (tuned)", "LightGBM"); if (!(m.key in state.on)) state.on[m.key] = !!m.default || ["xgboost", "random_forest"].includes(m.key); state.mode[m.key] = state.mode[m.key] || "hmm"; });
      $("connDot").className = "dot on";
      renderTrips(); renderModels(); toast(`Connected · ${trips.length} journeys · ${models.length} models`);
      if (!state.sel && trips.length && !qs.get("trip")) selectTrip(trips[0].trip_id);
      const h = document.querySelector("#hero ol"); if (h) h.innerHTML = `<li><b>Connected</b> ✓ ${trips.length} evaluation journeys loaded.</li><li><b>Choose</b> a journey on the left (the first one is selected).</li><li>Press <b>Reconstruct route</b> to compare the models.</li>`;
    } catch (e) {
      $("connDot").className = "dot err"; toast("Could not reach the server — check the IP, port and Wi-Fi");
    }
  }

  function renderTrips() {
    $("tripCount").textContent = state.trips.length ? `${state.trips.length} trips` : "";
    $("trips").innerHTML = state.trips.map((t, i) =>
      `<div class="trip ${state.sel === t.trip_id ? "sel" : ""}" data-id="${t.trip_id}">
        <div class="ix">${String(i + 1).padStart(2, "0")}</div>
        <div><div class="tid">${t.trip_id}</div><div class="meta">${fmtDur(t.duration_s)} · ${((t.bbox[0] + t.bbox[2]) / 2).toFixed(2)}°, ${((t.bbox[1] + t.bbox[3]) / 2).toFixed(2)}°</div></div>
        <div class="pts">${t.points.toLocaleString()}<div class="meta">points</div></div>
      </div>`).join("") || `<div class="placeholder">The server has no trips in its data folder.</div>`;
    document.querySelectorAll(".trip").forEach((el) => el.onclick = () => selectTrip(el.dataset.id));
    updateRunBtn();
  }

  function selectTrip(id) {
    state.sel = id; renderTrips();
    const t = state.trips.find((x) => x.trip_id === id);
    if (t) { state.bounds = L.latLngBounds([[t.bbox[0], t.bbox[1]], [t.bbox[2], t.bbox[3]]]); fitTrip(1.6); }
  }

  function renderModels() {
    $("models").innerHTML = state.models.map((m) => {
      const st = m.student_test || {};
      return `<div class="model ${state.on[m.key] ? "" : "off"}" data-k="${m.key}" style="--c:${m.c}">
        <div class="sw" style="background:${m.c}" data-act="tog"></div>
        <div class="ml" data-act="tog"><b>${m.label}${m.default ? '<span class="def">default</span>' : ""}</b><span>dev-test · raw ${pct(st.top1)}% → HMM ${pct(st.hmm_pointwise)}%</span></div>
        <div class="seg"><button data-act="hmm" class="${state.mode[m.key] === "hmm" ? "on" : ""}">HMM</button><button data-act="clf" class="${state.mode[m.key] === "clf" ? "on" : ""}">Raw</button></div>
      </div>`;
    }).join("");
    document.querySelectorAll(".model").forEach((el) => el.onclick = (e) => {
      const k = el.dataset.k, act = e.target.dataset.act; if (!act) return;
      if (act === "tog") state.on[k] = !state.on[k]; else state.mode[k] = act;
      renderModels(); updateRunBtn();
      if (state.result && act !== "tog") { swapRoutes(false); renderBoard(); buildCursor(); }
      else if (state.result && act === "tog") { const have = state.result.models.map((x) => x.key); if (state.on[k] && !have.includes(k)) run(); else { swapRoutes(false); renderBoard(); buildCursor(); } }
    });
  }

  function pads() {
    const small = innerWidth < 860;
    const left = !small && !document.body.classList.contains("hide-left") ? 380 : 40;
    const right = !small && !document.body.classList.contains("hide-right") && !$("results").classList.contains("hidden") ? 400 : 40;
    const bottom = $("timeline").classList.contains("hidden") ? 40 : 140;
    return { paddingTopLeft: [left, 110], paddingBottomRight: [right, bottom] };
  }
  function fitTrip(dur) { if (state.bounds) map.flyToBounds(state.bounds, { ...pads(), duration: dur || 1.2, easeLinearity: 0.15 }); }
  window.__fit = fitTrip;
  const activeKeys = () => state.models.filter((m) => state.on[m.key]).map((m) => m.key);
  function updateRunBtn() {
    const ok = !!(state.sel && activeKeys().length); const b = $("runBtn"); b.disabled = !ok;
    b.querySelector("span").textContent = !state.trips.length ? "Connect to the server first" : !state.sel ? "Choose a journey first" : !activeKeys().length ? "Turn on at least one model" : `Reconstruct route · ${state.sel}`;
  }

  // ---------- run ----------
  async function run() {
    const keys = activeKeys(); if (!state.sel || !keys.length) return;
    stop(); $("hero").classList.add("hidden"); $("loader").classList.remove("hidden"); $("ldTrip").textContent = state.sel;
    const lis = [...document.querySelectorAll("#ldSteps li")]; lis.forEach((l) => l.classList.remove("on"));
    let s = 0; const tick = setInterval(() => { if (s < lis.length) lis[s++].classList.add("on"); }, 380);
    try {
      const res = await call("POST", "/predict", { trip_id: state.sel, models: keys });
      res.models.forEach((m) => { const meta = state.models.find((x) => x.key === m.key); m.c = meta ? meta.c : m.colour; if (meta) m.label = meta.label; });
      ["net", "gps", "routes", "cursor"].forEach((p) => map.getPane(p).style.opacity = 0);
      await new Promise((ok) => setTimeout(ok, 250));
      state.result = res; state.idx = 0; state.pos = 0; lastChipIdx = -1;
      drawBase(); drawRoutes(ANIM); renderSummary(); setupTimeline();
      ["net", "gps", "routes", "cursor"].forEach((p) => map.getPane(p).style.opacity = 1);
      $("results").classList.remove("hidden"); $("timeline").classList.remove("hidden"); document.body.classList.add("has-result");
      requestAnimationFrame(() => { $("results").classList.add("in"); $("timeline").classList.add("in"); renderBoard(); });
    } catch (e) { toast("Prediction failed: " + e.message.slice(0, 120)); }
    finally { clearInterval(tick); lis.forEach((l) => l.classList.add("on")); setTimeout(() => $("loader").classList.add("hidden"), 200); }
  }

  function drawBase() {
    const r = state.result; G.net.clearLayers(); G.gps.clearLayers();
    L.polyline(r.network.map((s) => [[s[0], s[1]], [s[2], s[3]]]), { renderer: R.net, color: "#7d8597", weight: 1.2, opacity: 0.35, interactive: false }).addTo(G.net);
    const step = Math.max(1, Math.ceil(r.gps.length / 3000));
    r.gps.forEach((g, i) => { if (i % step === 0) L.circleMarker([g[0], g[1]], { renderer: R.gps, radius: 2.2, stroke: false, fill: true, fillColor: "#1f2a44", fillOpacity: 0.28, interactive: false }).addTo(G.gps); });
    const a = r.gps[0], b = r.gps[r.gps.length - 1];
    if (a) { L.marker([a[0], a[1]], { pane: "cursor", icon: L.divIcon({ className: "", html: '<div class="pin">Start</div>', iconSize: null, iconAnchor: [-8, 12] }) }).addTo(G.net);
             L.marker([b[0], b[1]], { pane: "cursor", icon: L.divIcon({ className: "", html: '<div class="pin end">End</div>', iconSize: null, iconAnchor: [-8, 12] }) }).addTo(G.net); }
    const lats = r.gps.map((g) => g[0]), lons = r.gps.map((g) => g[1]);
    state.bounds = L.latLngBounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]]);
    fitTrip(1.4);
  }

  const near = (a, b) => Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6;
  function mergeLines(lines) {          // join touching fragments into long smooth polylines
    const out = [];
    lines.forEach((pl) => { if (!pl.length) return; const last = out[out.length - 1];
      if (last && near(last[last.length - 1], pl[0])) last.push(...pl.slice(1)); else out.push(pl.slice()); });
    return out;
  }
  function swapRoutes(animate) {        // cross-fade old routes out, new routes in
    const pane = map.getPane("routes");
    if (!state.result || !G.routes.getLayers().length) return drawRoutes(animate);
    pane.style.opacity = 0;
    setTimeout(() => { drawRoutes(animate); pane.style.opacity = 1; }, 260);
  }
  function drawRoutes(animate) {
    G.routes.clearLayers(); const r = state.result; if (!r) return;
    const shown = r.models.filter((m) => state.on[m.key]); const n = shown.length;
    // nested bands, all centred on the road: widest first, thinnest on top
    const widths = n === 1 ? [5] : Array.from({ length: n }, (_, i) => 3 + (n - 1 - i) * 4);
    const plan = shown.map((m, i) => ({ m, i, mode: state.mode[m.key], lines: mergeLines(state.mode[m.key] === "hmm" ? m.hmm_route : m.clf_route), w: widths[i] }));
    plan.slice(0, 1).forEach(({ lines, w }) => lines.forEach((pl) =>
      L.polyline(pl, { renderer: R.routes, color: "#fff", weight: w + 4, opacity: 0.95, interactive: false, lineJoin: "round", lineCap: "round" }).addTo(G.routes)));
    plan.forEach(({ m, i, mode, lines, w }) => {
      lines.forEach((pl) => {
        const ln = L.polyline(pl, { renderer: R.routes, color: m.c, weight: w, opacity: 1, className: "route-path", dashArray: mode === "clf" ? "1 " + (w + 5) : null, lineCap: "round", lineJoin: "round", smoothFactor: 1.2 })
          .bindTooltip(`${m.label} · ${mode === "hmm" ? "HMM" : "raw classifier"}`, { sticky: true }).addTo(G.routes);
        if (animate && mode === "hmm") {
          const el = ln.getElement(); if (!el) return;
          const len = el.getTotalLength(); el.style.transition = "none"; el.style.strokeDasharray = `${len} ${len}`; el.style.strokeDashoffset = len;
          el.getBoundingClientRect();
          setTimeout(() => { el.style.transition = `stroke-dashoffset 2.2s cubic-bezier(.45,.05,.25,1) ${i * 0.3}s`; el.style.strokeDashoffset = 0; }, 900);
          setTimeout(() => { el.style.transition = ""; el.style.strokeDasharray = ""; }, 900 + 2200 + i * 300 + 150);
        }
      });
    });
  }

  function renderSummary() {
    const r = state.result; const dur = r.gps.length ? r.gps[r.gps.length - 1][2] - r.gps[0][2] : 0;
    $("tripTag").textContent = "#" + r.trip_id;
    $("summary").innerHTML = [[r.points.toLocaleString(), "GPS points"], [fmtDur(dur), "duration"], [(r.total_ms / 1000).toFixed(2) + "s", "server time"]]
      .map(([v, k]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  }

  function lenBar(x) {                  // stacked bar: correct / wrongly added / missed length (numbers only, no geometry)
    if (x.ref_length_km == null || x.length_added_m == null) return "";
    const ref = x.ref_length_km * 1000, add = x.length_added_m, miss = x.length_missed_m, ok = Math.max(0, ref - miss);
    const tot = ok + miss + add || 1, w = (v) => (100 * v / tot).toFixed(2) + "%";
    return `<div class="lbar" title="Length error = (added + missed) / true route length">
      <div class="lbar-track"><i class="ok" style="width:${w(ok)}"></i><i class="miss" style="width:${w(miss)}"></i><i class="add" style="width:${w(add)}"></i></div>
      <div class="lbar-key"><span><i class="ok"></i>correct ${(ok / 1000).toFixed(2)} km</span><span><i class="miss"></i>missed ${Math.round(miss)} m</span><span><i class="add"></i>added ${Math.round(add)} m</span></div>
    </div>`;
  }

  function renderBoard() {
    const r = state.result; if (!r) return;
    const shown = r.models.filter((m) => state.on[m.key]);
    const acc = (m) => (state.mode[m.key] === "hmm" ? m.metrics.accuracy_hmm : m.metrics.accuracy_clf);
    shown.sort((a, b) => (acc(b) || 0) - (acc(a) || 0));
    $("board").innerHTML = shown.map((m, i) => {
      const x = m.metrics, lift = (x.accuracy_hmm - x.accuracy_clf) * 100, hmm = state.mode[m.key] === "hmm";
      const rows = [["Raw classifier", x.accuracy_clf, true], ["+ HMM", x.accuracy_hmm], ["Route F1", hmm ? x.route_f1 : x.route_f1_clf], ["Continuity", hmm ? x.continuity_hmm : x.continuity_clf]];
      return `<div class="card" style="--c:${m.c}; animation-delay:${i * 80}ms">
        <div class="row1"><span class="rank">#${i + 1}</span><span class="lbl">${m.label}</span><span class="lift ${lift < 0 ? "neg" : ""}">${lift >= 0 ? "+" : ""}${lift.toFixed(1)} pts from HMM</span></div>
        <div class="big">${pct(acc(m))}%</div><div class="cap">points on the right road · ${hmm ? "HMM" : "raw classifier"}</div>
        <div class="bars">${rows.map(([k, v, soft]) => `<span>${k}</span><div class="bar"><i class="${soft ? "soft" : ""}" data-w="${(100 * (v || 0)).toFixed(1)}"></i></div><span class="n">${pct(v)}</span>`).join("")}</div>
        <div class="lenerr">
          <div><span>Length error</span><b>${pct(hmm ? x.length_error : x.length_error_clf)}%</b><small>lower is better</small></div>
          <div><span>Route length</span><b>${x.pred_length_km != null ? x.pred_length_km.toFixed(2) : "–"} km</b><small>${x.length_diff_pct != null ? (x.length_diff_pct >= 0 ? "+" : "") + (100 * x.length_diff_pct).toFixed(1) + "% vs true" : ""}</small></div>
          <div><span>Added / missed</span><b>${x.length_added_m != null ? Math.round(x.length_added_m) : "–"} / ${x.length_missed_m != null ? Math.round(x.length_missed_m) : "–"} m</b><small>metres, HMM</small></div>
        </div>
        ${lenBar(x)}
        <div class="foot"><span>${m.n_route_arcs} road segments</span><span>${Math.round(m.infer_ms)} ms model time</span></div>
      </div>`;
    }).join("") || `<div class="placeholder">Turn on at least one model.</div>`;
    // grow bars from a min of 80% so differences are visible
    requestAnimationFrame(() => document.querySelectorAll(".bar i").forEach((b) => { const w = parseFloat(b.dataset.w); b.style.width = Math.max(2, ((w - 80) / 20) * 100) + "%"; }));
    countUp();
  }

  function countUp() {                 // numbers roll up smoothly instead of popping in
    document.querySelectorAll(".card .big").forEach((el) => {
      const target = parseFloat(el.textContent); if (!isFinite(target)) return;
      const t0 = performance.now(), from = 80;
      const step = (t) => { const k = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - k, 3);
        el.textContent = (from + (target - from) * e).toFixed(1) + "%"; if (k < 1) requestAnimationFrame(step); };
      requestAnimationFrame(step);
    });
  }

  // ---------- timeline ----------
  const C = { gps: null, items: [] };
  function setupTimeline() {
    const n = state.result.gps.length; const s = $("scrub"); s.max = Math.max(0, n - 1); s.value = 0; state.pos = 0; buildCursor();
  }
  function buildCursor() {               // create cursor layers once; later frames only move them
    G.cursor.clearLayers(); C.items = []; const r = state.result; if (!r || !r.gps.length) return;
    r.models.filter((m) => state.on[m.key]).forEach((m) => {
      const line = L.polyline([[0, 0], [0, 0]], { renderer: R.cursor, color: m.c, weight: 2, dashArray: "3 4", interactive: false }).addTo(G.cursor);
      const dot = L.circleMarker([0, 0], { renderer: R.cursor, radius: 5.5, color: "#fff", weight: 2, fillColor: m.c, fillOpacity: 1, interactive: false }).addTo(G.cursor);
      C.items.push({ m, line, dot });
    });
    C.gps = L.marker([0, 0], { pane: "cursor", icon: L.divIcon({ className: "", html: '<div class="gps-cursor"></div>', iconSize: [18, 18], iconAnchor: [9, 9] }), interactive: false }).addTo(G.cursor);
    cursor(state.pos || state.idx || 0, true);
  }
  const lerp = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
  let lastChipIdx = -1;
  function cursor(pos, force) {          // pos may be fractional: we interpolate between GPS points
    const r = state.result; if (!r || !r.gps.length || !C.gps) return;
    pos = Math.max(0, Math.min(r.gps.length - 1, pos == null ? state.idx : pos)); state.pos = pos;
    const i = Math.floor(pos), j = Math.min(r.gps.length - 1, i + 1), k = pos - i;
    const g = lerp(r.gps[i], r.gps[j], k);
    C.gps.setLatLng(g);
    const arcs = [];
    C.items.forEach(({ m, line, dot }) => {
      const arr = state.mode[m.key] === "hmm" ? m.matched_hmm : m.matched_clf; const a = arr[i], b = arr[j] || a;
      if (a) { const p = b ? lerp(a, b, k) : a; dot.setLatLng(p); line.setLatLngs([g, p]); dot.setStyle({ opacity: 1, fillOpacity: 1 }); line.setStyle({ opacity: 1 }); arcs.push(a[2]); }
      else { dot.setStyle({ opacity: 0, fillOpacity: 0 }); line.setStyle({ opacity: 0 }); }
    });
    $("scrub").style.setProperty("--p", (100 * pos / Math.max(1, r.gps.length - 1)) + "%");
    if (i === lastChipIdx && !force) return; lastChipIdx = i;
    $("clock").textContent = fmtDur(r.gps[i][2] - r.gps[0][2]); $("ptno").textContent = `point ${(i + 1).toLocaleString()} / ${r.gps.length.toLocaleString()}`;
    $("chips").innerHTML = C.items.map(({ m }) => { const a = (state.mode[m.key] === "hmm" ? m.matched_hmm : m.matched_clf)[i];
      return `<span class="chip" style="--c:${m.c}"><i></i>${m.label.split(" ")[0]} · ${a ? "road " + a[2] : "no road"}</span>`; }).join("");
    const ag = $("agree");
    if (arcs.length > 1) { const same = arcs.every((a) => a === arcs[0]); ag.className = "agree " + (same ? "yes" : "no"); ag.textContent = same ? "All models agree" : "Models disagree here"; }
    else { ag.className = "agree"; ag.textContent = ""; }
  }
  function play() {
    if (state.playing) return stop();
    const r = state.result; if (!r) return;
    if (state.pos >= r.gps.length - 1) state.pos = 0;
    $("playIcon").innerHTML = '<path d="M5 4h4v12H5zM11 4h4v12h-4z" fill="currentColor"/>';
    let last = performance.now();
    const frame = (t) => {             // ~speed GPS points per 60 ms, interpolated every animation frame
      if (!state.playing) return;
      const dt = t - last; last = t;
      const pos = Math.min(r.gps.length - 1, state.pos + (dt / 60) * parseInt($("speed").value, 10));
      state.idx = Math.floor(pos); $("scrub").value = state.idx; cursor(pos);
      if (pos >= r.gps.length - 1) return stop();
      if (follow.checked) map.panTo(C.gps.getLatLng(), { animate: false });
      state.playing = requestAnimationFrame(frame);
    };
    state.playing = requestAnimationFrame(frame);
  }
  function stop() { if (state.playing) cancelAnimationFrame(state.playing); state.playing = null; $("playIcon").innerHTML = '<path d="M6 4l10 6-10 6z" fill="currentColor"/>'; }
  const follow = { checked: false };

  // ---------- wiring ----------
  $("connectBtn").onclick = connect;
  $("apiUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  $("runBtn").onclick = run;
  $("playBtn").onclick = play;
  $("scrub").oninput = (e) => { stop(); glideTo(parseInt(e.target.value, 10)); };
  function glideTo(target) {             // dragging / arrow keys ease the cursor instead of jumping
    state.idx = target; const from = state.pos || 0, t0 = performance.now();
    cancelAnimationFrame(glideTo.h);
    const step = (t) => { const k = Math.min(1, (t - t0) / 180); cursor(from + (target - from) * (1 - Math.pow(1 - k, 3))); if (k < 1) glideTo.h = requestAnimationFrame(step); };
    glideTo.h = requestAnimationFrame(step);
  }
  const toggleSide = (side) => { document.body.classList.toggle("hide-" + side); setTimeout(() => { map.invalidateSize({ pan: false }); fitTrip(0.9); }, 420); };
  $("hideLeft").onclick = () => toggleSide("left");
  $("hideRight").onclick = () => toggleSide("right");
  $("fitBtn").onclick = () => fitTrip(1.0);
  $("logBtn").onclick = () => $("logDrawer").classList.toggle("hidden");
  $("logClose").onclick = () => $("logDrawer").classList.add("hidden");
  document.addEventListener("keydown", (e) => {
    if (!state.result || e.target.tagName === "INPUT") return;
    if (e.key === " ") { e.preventDefault(); play(); }
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") { stop(); const t = Math.max(0, Math.min(state.result.gps.length - 1, state.idx + (e.key === "ArrowRight" ? 10 : -10))); $("scrub").value = t; glideTo(t); }
  });

  $("apiUrl").value = qs.get("api") || store.get("pathsnap.api") || "http://localhost:8000";
  if (qs.get("api") || store.get("pathsnap.api")) connect().then(() => {
    if (qs.get("trip")) { selectTrip(qs.get("trip")); if (qs.get("run") === "1") run().then(() => { const t = parseInt(qs.get("t") || "0", 10); if (t && state.result) { state.idx = Math.min(t, state.result.gps.length - 1); $("scrub").value = state.idx; cursor(state.idx, true); } }); }
  });
})();
