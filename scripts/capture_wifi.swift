// Read the current Wi-Fi link only. No scanning, SSIDs, BSSIDs or credentials.
import Foundation
import CoreWLAN

func argument(_ flag: String, fallback: Double) -> Double {
    guard let index = CommandLine.arguments.firstIndex(of: flag),
          index + 1 < CommandLine.arguments.count,
          let value = Double(CommandLine.arguments[index + 1]) else { return fallback }
    return value
}
let duration = argument("--duration", fallback: 60)
let interval = argument("--interval", fallback: 0.5)
guard duration.isFinite && duration >= 5 && duration <= 300,
      interval.isFinite && interval >= 0.2 && interval <= 2 else {
    fputs("Use duration 5–300 seconds and interval 0.2–2 seconds.\n", stderr)
    exit(2)
}
guard let wifi = CWWiFiClient.shared().interface(), wifi.powerOn() else {
    fputs("No powered Wi-Fi interface. Connect this Mac to your router first.\n", stderr)
    exit(2)
}
setbuf(stdout, nil)
print("# source=macos-corewlan-rssi")
print("# Measurements describe link quality; human presence is not inferred.")
print("time_s,rssi_dbm,noise_dbm,channel,status")
let start = ProcessInfo.processInfo.systemUptime
var count = 0
var validCount = 0
while ProcessInfo.processInfo.systemUptime - start < duration {
    let now = ProcessInfo.processInfo.systemUptime - start
    let rssi = wifi.rssiValue()
    let noise = wifi.noiseMeasurement()
    let channel = wifi.wlanChannel()?.channelNumber ?? 0
    let valid = wifi.powerOn() && rssi >= -127 && rssi < 0
    let noiseText = noise >= -127 && noise < 0 ? String(noise) : ""
    print(String(format: "%.6f", now) + "," + (valid ? String(rssi) : "")
          + "," + noiseText + "," + String(channel) + "," + (valid ? "connected" : "unavailable"))
    if valid { validCount += 1 }
    count += 1
    let delay = start + Double(count) * interval - ProcessInfo.processInfo.systemUptime
    if delay > 0 { Thread.sleep(forTimeInterval: delay) }
}
if validCount == 0 {
    fputs("The Wi-Fi interface returned no RSSI measurements. Check the connection and OS restrictions.\n", stderr)
    exit(3)
}
