# PathSnap UI: how to run it on your laptop (Laptop 2)

This folder is the whole UI: `index.html`, `app.js` and `style.css`. You don't need Python packages, the models or the data. Everything heavy runs on Nikitha's Mac (Laptop 1). Your laptop only needs a browser and the **internet**, because the map tiles and fonts load online.

## Before the evaluation (5 minutes, do it once to test)

1. **Unzip** `PathSnap_UI.zip` anywhere, for example on the Desktop.
2. **Join the same network as Nikitha's Mac.** College Wi-Fi often blocks laptop-to-laptop connections. If it does, Nikitha turns on her phone hotspot and **both** laptops join it.
3. **Get the API address from Nikitha.** She runs `ipconfig getifaddr en0` on her Mac and tells you the number, for example `10.1.24.165`. Your API address is then:
   ```
   http://10.1.24.165:8000
   ```
   The number changes if you change Wi-Fi or hotspot, so ask again after switching.
4. **Quick connection test:** in your browser open `http://<that-IP>:8000/health`.
   You should see `{"status":"ok","models":[...],"trips":10}`.
   If the page doesn't load, see Troubleshooting below.
5. **Start the UI.** Open a terminal inside the unzipped `ui` folder and run:
   ```
   python3 -m http.server 5500
   ```
   On Windows, use `python -m http.server 5500`.
   Then open **http://localhost:5500** in Chrome.
   If you have no Python, double-clicking `index.html` also works.
6. **Connect.** Type the API address into the box at the top right and click **Connect**. The dot turns **green**, and the 10 evaluation journeys appear on the left.

## During the demo

1. **Step 1, Choose a journey:** click any of the 10 trips. The map flies to it.
2. **Step 2, Pick models:** click a model's name or colour square to turn it on or off. Ensemble is the default. Keep 2–3 on.
   - **HMM** = the final connected route.
   - **Raw** = the classifier only. Use it to show why the HMM matters.
3. Click **Reconstruct route**. The coloured routes draw themselves along the roads, and the **Results** panel shows each model's accuracy, which the server computes.
4. Press **▶**, or the **space bar**, to replay the drive. The marker shows which road each model snapped the GPS point to, plus "All models agree" or "Models disagree here". The ← → keys step through the drive.
5. **Useful buttons:**
   - **Fit trip:** shows the whole route.
   - The **arrows** beside the panels hide them, for a full-screen map.
   - The **layers icon** (bottom right) switches to Satellite / Topographic / OpenStreetMap.
   - **API log:** shows each request this page sent.
6. Point the evaluator to **Nikitha's terminal**. It prints a log line for every request, which the rules ask us to show.

## Troubleshooting

| Problem | Fix |
|---|---|
| Dot turns red / "Could not reach the server" | The IP is wrong, the server isn't running, or you're on different networks. Re-check the IP, then try `http://<IP>:8000/health`. |
| `/health` doesn't load but both laptops are on the same Wi-Fi | The college Wi-Fi is blocking it. Switch both laptops to Nikitha's phone hotspot, and get the new IP. |
| Still blocked on the hotspot | On Nikitha's Mac: System Settings → Network → Firewall → allow Python (or turn the firewall off for the demo). |
| The map is grey / blank | Your laptop has no internet. The routes still work, but the map tiles need internet. |
| Changes in the UI don't show | Hard refresh with Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows). |
