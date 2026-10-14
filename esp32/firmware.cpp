// HOW TO USE: this is a single, self-contained file. Open the Arduino IDE, create a new
// sketch. paste this whole file in, replace wifi ssid and password.
// (Only the ESP32 core's WiFi.h / WiFiUdp.h / WebServer.h are needed.)

#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>

const char* ssid = "ssid_here"
const char* password = "password_here";

// Network Ports & Servers
const uint16_t TCP_PORT = 5000;
const uint16_t UDP_DISCOVERY_PORT = 5001;

WiFiServer tcpServer(TCP_PORT);
WiFiUDP udp;
WebServer server(80);

// Pin mapping for outputs 1 through 8
const int outputPins[8] = {32, 33, 25, 26, 27, 18, 19, 23};

// --- non-blocking TCP client pool ---
const int MAX_CLIENTS = 4;                  // simultaneous control connections to serve
const unsigned long CLIENT_IDLE_MS = 5000;  // drop a silent/stale client after this long
const size_t MAX_LINE = 64;                 // longest accepted command line (garbage guard)

struct TcpConn {
  WiFiClient client;
  String buf;
  unsigned long lastMs = 0;
  bool active = false;
};
TcpConn conns[MAX_CLIENTS];

// Non-blocking "MOM" (momentary) pulses for the web UI: pin index -> millis() deadline.
unsigned long momOffAt[8] = {0};

// Forward declarations so the Arduino IDE's auto-generated prototypes are always harmless:
// every signature below uses ONLY primitive / core-library types (no sketch-defined types),
// so they can be inserted anywhere without an "undeclared type" error.
void handleRoot();
void handleStatus();
void handleCommand();
void applySet(int slot, int channel, int state);
void processLine(int slot, String line);
void serviceTcp();
void serviceUdp();
void serviceMom();
void setup();
void loop();

// HTML Page served on Port 80
const char MAIN_PAGE[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32 GPIO Control</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; display: flex; justify-content: center; }
    .container { width: 100%; max-width: 480px; }
    h2 { text-align: center; color: #38bdf8; margin-bottom: 24px; }
    .card { background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); display: flex; align-items: center; justify-content: space-between; }
    .info { display: flex; align-items: center; gap: 12px; }
    .badge { width: 12px; height: 12px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 8px #ef4444; transition: 0.1s; }
    .badge.active { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
    .label { font-weight: 600; font-size: 16px; }
    .buttons { display: flex; gap: 6px; }
    button { border: none; border-radius: 6px; padding: 8px 12px; font-weight: 600; cursor: pointer; color: white; transition: filter 0.1s; }
    button:active { filter: brightness(0.85); }
    .btn-on { background: #16a34a; }
    .btn-off { background: #dc2626; }
    .btn-mom { background: #ca8a04; }
  </style>
</head>
<body>
  <div class="container">
    <h2>ESP32 GPIO Controller</h2>
    <div id="channels"></div>
  </div>

<script>
    let isFetching = false;

    function buildUI() {
      const container = document.getElementById('channels');
      let html = '';
      for (let i = 1; i <= 8; i++) {
        html += `
          <div class="card">
            <div class="info">
              <div id="badge-${i}" class="badge"></div>
              <div class="label">Out ${i}</div>
            </div>
            <div class="buttons">
              <button class="btn-on" onclick="sendCmd(${i}, 'on')">ON</button>
              <button class="btn-off" onclick="sendCmd(${i}, 'off')">OFF</button>
              <button class="btn-mom" onclick="sendCmd(${i}, 'mom')">MOM</button>
            </div>
          </div>`;
      }
      container.innerHTML = html;
      updateStates();
      setInterval(updateStates, 250);
    }

    async function sendCmd(ch, action) {
      try {
        await fetch(`/cmd?ch=${ch}&action=${action}`);
        updateStates();
      } catch (err) {
        console.error('Command failed', err);
      }
    }

    async function updateStates() {
      if (isFetching) return;
      isFetching = true;
      try {
        const res = await fetch('/status');
        const data = await res.json();
        for (let i = 1; i <= 8; i++) {
          const badge = document.getElementById(`badge-${i}`);
          if (badge) {
            // State 1 = HIGH (ON), State 0 = LOW (OFF)
            if (data[i-1] === 1) {
              badge.classList.add('active');
            } else {
              badge.classList.remove('active');
            }
          }
        }
      } catch (err) {
        console.error('Status fetch failed', err);
      } finally {
        isFetching = false;
      }
    }

    window.onload = buildUI;
  </script>
</body>
</html>
)rawliteral";

void handleRoot() {
  server.send(200, "html", MAIN_PAGE);
}

void handleStatus() {
  String json = "[";
  for (int i = 0; i < 8; i++) {
    int pinState = digitalRead(outputPins[i]);
    json += (pinState == HIGH) ? "1" : "0";
    if (i < 7) json += ",";
  }
  json += "]";
  server.send(200, "application/json", json);
}

void handleCommand() {
  if (!server.hasArg("ch") || !server.hasArg("action")) {
    server.send(400, "text/plain", "BAD_ARGS");
    return;
  }

  int channel = server.arg("ch").toInt();
  String action = server.arg("action");

  if (channel < 1 || channel > 8) {
    server.send(400, "text/plain", "BAD_CHANNEL");
    return;
  }

  int pin = outputPins[channel - 1];

  if (action == "on") {
    momOffAt[channel - 1] = 0;
    digitalWrite(pin, HIGH); // Active High (ON)
    server.send(200, "text/plain", "OK");
  }
  else if (action == "off") {
    momOffAt[channel - 1] = 0;
    digitalWrite(pin, LOW);  // Inactive Low (OFF)
    server.send(200, "text/plain", "OK");
  }
  else if (action == "mom") {
    digitalWrite(pin, HIGH);                    // Turn ON
    momOffAt[channel - 1] = millis() + 500;     // schedule OFF in 500ms (non-blocking)
    server.send(200, "text/plain", "OK");
  }
  else {
    server.send(400, "text/plain", "BAD_ACTION");
  }
}

void applySet(int slot, int channel, int state) {
  if (channel < 1 || channel > 8) {
    conns[slot].client.println("ERR_CHANNEL");
    return;
  }
  momOffAt[channel - 1] = 0;   // a direct SET overrides any pending momentary pulse
  digitalWrite(outputPins[channel - 1], state ? HIGH : LOW);
  conns[slot].client.println("OK");
}

void processLine(int slot, String line) {
  if (line.startsWith("SET ")) {
    int channel = 0, state = 0;
    if (sscanf(line.c_str(), "SET %d %d", &channel, &state) == 2) {
      applySet(slot, channel, state);
    } else {
      conns[slot].client.println("ERR_FORMAT");
    }
  } else if (line == "PING") {
    conns[slot].client.println("PONG");
  } else {
    conns[slot].client.println("ERR_CMD");
  }
}

// Non-blocking: accept new connections and service every active one a little each pass.
void serviceTcp() {
  unsigned long now = millis();

  // Accept at most one new client per pass; more are picked up on later passes.
  WiFiClient nc = tcpServer.available();
  if (nc) {
    int slot = -1;
    for (int i = 0; i < MAX_CLIENTS; i++) {
      if (!conns[i].active) { slot = i; break; }
    }
    if (slot >= 0) {
      conns[slot].client = nc;
      conns[slot].client.setNoDelay(true);
      conns[slot].client.setTimeout(0);   // non-blocking reads
      conns[slot].buf = "";
      conns[slot].lastMs = now;
      conns[slot].active = true;
    } else {
      // Pool full: answer immediately instead of queueing forever.
      nc.println("ERR_BUSY");
      nc.stop();
    }
  }

  for (int i = 0; i < MAX_CLIENTS; i++) {
    if (!conns[i].active) continue;
    WiFiClient& c = conns[i].client;

    if (!c.connected()) {
      c.stop();
      conns[i].active = false;
      conns[i].buf = "";
      continue;
    }

    bool activity = false;
    int budget = 512;   // cap bytes handled per pass so one noisy client can't hog the loop
    while (budget-- > 0 && c.available()) {
      int ch = c.read();
      if (ch < 0) break;
      activity = true;
      if (ch == '\n') {
        processLine(i, conns[i].buf);
        conns[i].buf = "";
      } else if (ch != '\r') {
        if (conns[i].buf.length() < MAX_LINE) conns[i].buf += (char)ch;
        else conns[i].buf = "";   // discard over-long garbage instead of growing memory
      }
    }
    if (activity) conns[i].lastMs = now;

    // Drop silent/stale clients so one stuck connection can never wedge the controller.
    if (now - conns[i].lastMs > CLIENT_IDLE_MS) {
      c.stop();
      conns[i].active = false;
      conns[i].buf = "";
    }
  }
}

void serviceUdp() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char incomingPacket[256];
    int len = udp.read(incomingPacket, sizeof(incomingPacket) - 1);
    if (len > 0) {
      incomingPacket[len] = '\0';
    } else {
      incomingPacket[0] = '\0';
    }

    if (String(incomingPacket).indexOf("DISCOVER_ESP32") >= 0) {
      udp.beginPacket(udp.remoteIP(), udp.remotePort());
      udp.print("ESP32_ACK");
      udp.endPacket();
    }
  }
}

// Turn off any momentary pulse whose 500ms window has elapsed.
void serviceMom() {
  unsigned long now = millis();
  for (int i = 0; i < 8; i++) {
    if (momOffAt[i] != 0 && now >= momOffAt[i]) {
      digitalWrite(outputPins[i], LOW);
      momOffAt[i] = 0;
    }
  }
}

void setup() {
  Serial.begin(115200);

  // Initialize all pins to OUTPUT and LOW (Off state)
  for (int i = 0; i < 8; i++) {
    pinMode(outputPins[i], OUTPUT);
    digitalWrite(outputPins[i], LOW);
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // keep the radio awake for low, consistent latency
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 30000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi not connected yet; will keep retrying in loop().");
  }

  tcpServer.begin();
  tcpServer.setNoDelay(true);
  udp.begin(UDP_DISCOVERY_PORT);

  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  server.on("/cmd", handleCommand);
  server.begin();
  Serial.println("Ready. TCP:5000  UDP:5001  HTTP:80");
}

void loop() {
  server.handleClient();
  serviceUdp();
  serviceTcp();
  serviceMom();

  // Keep WiFi alive: reconnect if it ever drops.
  static unsigned long lastWifiCheck = 0;
  if (millis() - lastWifiCheck > 5000) {
    lastWifiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi lost, reconnecting...");
      WiFi.disconnect();
      WiFi.begin(ssid, password);
    }
  }

  delay(1);   // yield to the WiFi/RTOS tasks so the stack stays responsive
}
