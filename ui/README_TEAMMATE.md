# PathSnap UI: how to run it on your laptop (Laptop 2)

This folder is the whole UI. It does not need any other file from the project. It only needs the address of the API running on Nikitha's Mac.

1. **Join the same Wi-Fi as Nikitha's Mac.** If the college Wi-Fi blocks laptop-to-laptop traffic, both of you join the same phone hotspot instead.
2. **Get the API address from Nikitha.** She runs `ipconfig getifaddr en0` on her Mac, which shows an address like `192.168.1.23`. The API address is then `http://192.168.1.23:8000`.
3. **Start the UI.** Open a terminal inside this `ui` folder and run:
   ```
   python3 -m http.server 5500
   ```
   No Python? Double-click `index.html` instead. That works too.
4. **Open the browser** at `http://localhost:5500`. Type the API address into the box at the top right and click **Connect**. The dot turns green and the 10 journeys appear on the left.
5. **Demo.** Click a journey, keep the 3 models selected, and click **Reconstruct route**. The routes draw in their colours. Press ▶ (or the space bar) to replay the drive and watch each model snap the GPS dot to a road. The **Findings** panel shows the accuracy, which the server computes. Nikitha's terminal shows the API log for every request. Show both screens to the evaluator.

Troubleshooting: if the dot turns red, check the IP address, check that the server is running, and check that macOS allowed incoming connections for Python on Nikitha's Mac.
