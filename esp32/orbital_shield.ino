/*
 * OrbitalShield — ESP32 Node
 * ===========================
 * Nó de telemetria GNSS para validação física indireta do OGII.
 *
 * Fluxo:
 *   1. Conecta ao WiFi e ao broker MQTT
 *   2. Assina orbitalshield/alerts  → recebe { "ogii": 82, "level": "CRÍTICO" }
 *   3. Atualiza máquina de estados (BAIXO / MODERADO / ALTO / CRÍTICO)
 *   4. Simula degradação GNSS proporcional ao nível de alerta
 *   5. Publica orbitalshield/esp32/telemetry a cada 5s
 *
 * Hipótese científica:
 *   Tempestades geomagnéticas degradam o sinal GNSS →
 *   HDOP sobe, satélites visíveis caem, fix piora.
 *   O ESP32 representa um receptor GNSS em campo (trator/drone agrícola).
 *
 * Dependências (Arduino Library Manager):
 *   - PubSubClient  by Nick O'Leary  >= 2.8
 *   - ArduinoJson   by Benoit Blanchon >= 6.21
 *
 * Broker padrão: test.mosquitto.org (demo)
 * Para produção: substituir por broker local (Mosquitto)
 *
 * Wokwi: https://wokwi.com/projects/new/esp32
 *   → Adicionar LEDs nos pinos 2 (verde), 4 (amarelo), 5 (vermelho)
 *   → Opcional: display I2C 128x64 para HDOP e satélites
 *
 * Nota: is_replay=true indica dados simulados — não confundir com medição real.
 */

 #include <WiFi.h>
 #include <PubSubClient.h>
 #include <ArduinoJson.h>
 
 // ─── Configuração WiFi ────────────────────────────────────────────────────────
 
 const char* WIFI_SSID     = "SEU_WIFI";        // substituir
 const char* WIFI_PASSWORD = "SUA_SENHA";       // substituir
 
 // ─── Configuração MQTT ────────────────────────────────────────────────────────
 
 const char* MQTT_BROKER   = "test.mosquitto.org";
 const int   MQTT_PORT     = 1883;
 const char* MQTT_CLIENT   = "orbital_esp32_01";
 
 // Tópicos
 const char* TOPIC_ALERTS    = "orbitalshield/alerts";
 const char* TOPIC_TELEMETRY = "orbitalshield/esp32/telemetry";
 
 // ─── Pinos dos LEDs de status ─────────────────────────────────────────────────
 
 const int LED_GREEN  = 2;   // BAIXO
 const int LED_YELLOW = 4;   // MODERADO / ALTO
 const int LED_RED    = 5;   // CRÍTICO
 
 // ─── Intervalo de publicação ──────────────────────────────────────────────────
 
 const unsigned long PUBLISH_INTERVAL_MS = 5000;   // 5 segundos
 
 // ─── Estado atual do sistema ──────────────────────────────────────────────────
 
 struct GnssState {
   float  hdop;
   int    satellites_visible;
   int    satellites_used;
   int    fix_quality;    // 0=no fix, 1=GPS, 2=DGPS, 3=RTK float, 4=RTK fixed
   String status;         // OK / DEGRADED / NO_RTK
 };
 
 // Alvos por nível de alerta (com ruído ±10% aplicado em runtime)
 struct AlertTarget {
   float  hdop_min, hdop_max;
   int    sat_min,  sat_max;
   int    fix_min,  fix_max;
   String status;
 };
 
 const AlertTarget TARGETS[] = {
   // BAIXO
   { 0.8f, 1.4f,  12, 16,  3, 4,  "OK"       },
   // MODERADO
   { 1.5f, 2.5f,   9, 12,  2, 3,  "DEGRADED" },
   // ALTO
   { 2.5f, 4.0f,   6,  9,  2, 3,  "DEGRADED" },
   // CRÍTICO
   { 4.0f, 8.0f,   3,  6,  1, 2,  "NO_RTK"   },
 };
 
 // Índice do alerta atual: 0=BAIXO, 1=MODERADO, 2=ALTO, 3=CRÍTICO
 int   currentAlertIdx  = 0;
 float currentOgii      = 50.0f;
 
 // Estado GNSS atual (interpolado suavemente)
 GnssState gnssState = { 1.0f, 14, 12, 4, "OK" };
 
 // Timestamp da última publicação
 unsigned long lastPublish = 0;
 
 // ─── Clientes ─────────────────────────────────────────────────────────────────
 
 WiFiClient   wifiClient;
 PubSubClient mqttClient(wifiClient);
 
 // ─── Funções auxiliares ───────────────────────────────────────────────────────
 
 /**
  * Mapeia string de nível para índice do array TARGETS.
  */
 int levelToIndex(const String& level) {
   if (level == "BAIXO")    return 0;
   if (level == "MODERADO") return 1;
   if (level == "ALTO")     return 2;
   if (level == "CRÍTICO")  return 3;
   return 0;
 }
 
 /**
  * Interpola suavemente um float em direção ao alvo.
  * Evita saltos abruptos de HDOP — mais realista fisicamente.
  * alpha: fator de suavização (0.0 = sem mudança, 1.0 = instantâneo)
  */
 float smoothStep(float current, float target, float alpha = 0.25f) {
   return current + alpha * (target - current);
 }
 
 /**
  * Adiciona ruído gaussiano simplificado (±range * 10%).
  */
 float addNoise(float value, float range) {
   float noise = ((float)random(-100, 101) / 1000.0f) * range;
   return value + noise;
 }
 
 /**
  * Atualiza gnssState interpolando em direção ao alvo do nível atual.
  * Chamada a cada loop — transição em ~3-5 publicações (15-25s).
  */
 void updateGnssState() {
   const AlertTarget& target = TARGETS[currentAlertIdx];
 
   // Alvo central de HDOP para o nível atual
   float hdopTarget = (target.hdop_min + target.hdop_max) / 2.0f;
   float satTarget  = (target.sat_min  + target.sat_max)  / 2.0f;
   float fixTarget  = (target.fix_min  + target.fix_max)  / 2.0f;
 
   // Interpolação suave
   gnssState.hdop               = smoothStep(gnssState.hdop, hdopTarget);
   gnssState.satellites_visible = (int)smoothStep(
       (float)gnssState.satellites_visible, satTarget
   );
   gnssState.satellites_used    = max(0, gnssState.satellites_visible - 2);
   gnssState.fix_quality        = (int)smoothStep(
       (float)gnssState.fix_quality, fixTarget
   );
   gnssState.status             = target.status;
 
   // Adiciona ruído realista
   gnssState.hdop = max(0.5f, addNoise(gnssState.hdop,
       target.hdop_max - target.hdop_min));
   gnssState.satellites_visible = max(0,
       gnssState.satellites_visible + random(-1, 2));
 }
 
 /**
  * Atualiza LEDs conforme o nível de alerta atual.
  */
 void updateLeds() {
   digitalWrite(LED_GREEN,  LOW);
   digitalWrite(LED_YELLOW, LOW);
   digitalWrite(LED_RED,    LOW);
 
   switch (currentAlertIdx) {
     case 0: digitalWrite(LED_GREEN,  HIGH); break;
     case 1: digitalWrite(LED_YELLOW, HIGH); break;
     case 2: digitalWrite(LED_YELLOW, HIGH); break;
     case 3: digitalWrite(LED_RED,    HIGH); break;
   }
 }
 
 /**
  * Publica telemetria GNSS atual no broker MQTT.
  * Payload compatível com db/models.py → Esp32Telemetry.
  */
 void publishTelemetry() {
   StaticJsonDocument<256> doc;
 
   doc["device_id"]          = MQTT_CLIENT;
   doc["hdop"]               = gnssState.hdop;
   doc["satellites_visible"] = gnssState.satellites_visible;
   doc["satellites_used"]    = gnssState.satellites_used;
   doc["fix_quality"]        = gnssState.fix_quality;
   doc["status"]             = gnssState.status;
   doc["ogii_received"]      = (int)currentOgii;
   doc["is_replay"]          = true;   // SEMPRE true — dados simulados
 
   // Coordenadas fixas (centro do Brasil agrícola — Goiás)
   doc["latitude"]           = -16.6869f;
   doc["longitude"]          = -49.2648f;
 
   char buffer[256];
   serializeJson(doc, buffer);
 
   bool ok = mqttClient.publish(TOPIC_TELEMETRY, buffer);
 
   Serial.print("[TELEMETRY] ");
   Serial.print(ok ? "OK" : "FALHOU");
   Serial.print(" | OGII=");
   Serial.print((int)currentOgii);
   Serial.print(" | HDOP=");
   Serial.print(gnssState.hdop, 2);
   Serial.print(" | Sats=");
   Serial.print(gnssState.satellites_visible);
   Serial.print(" | Fix=");
   Serial.print(gnssState.fix_quality);
   Serial.print(" | Status=");
   Serial.println(gnssState.status);
 }
 
 // ─── Callback MQTT ────────────────────────────────────────────────────────────
 
 /**
  * Chamado ao receber mensagem no tópico orbitalshield/alerts.
  * Payload esperado: { "ogii": 82, "level": "CRÍTICO" }
  */
 void onMqttMessage(char* topic, byte* payload, unsigned int length) {
   StaticJsonDocument<128> doc;
   DeserializationError err = deserializeJson(doc, payload, length);
 
   if (err) {
     Serial.print("[MQTT] JSON inválido: ");
     Serial.println(err.c_str());
     return;
   }
 
   float  newOgii  = doc["ogii"]  | currentOgii;
   String newLevel = doc["level"] | "BAIXO";
 
   currentOgii     = newOgii;
   currentAlertIdx = levelToIndex(newLevel);
 
   Serial.print("[ALERT] Recebido: OGII=");
   Serial.print((int)currentOgii);
   Serial.print(" | Level=");
   Serial.println(newLevel);
 
   updateLeds();
 }
 
 // ─── WiFi ─────────────────────────────────────────────────────────────────────
 
 void connectWifi() {
   Serial.print("[WiFi] Conectando a ");
   Serial.println(WIFI_SSID);
 
   WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
   int attempts = 0;
   while (WiFi.status() != WL_CONNECTED && attempts < 20) {
     delay(500);
     Serial.print(".");
     attempts++;
   }
 
   if (WiFi.status() == WL_CONNECTED) {
     Serial.println("\n[WiFi] Conectado!");
     Serial.print("[WiFi] IP: ");
     Serial.println(WiFi.localIP());
   } else {
     Serial.println("\n[WiFi] Falha na conexão. Reiniciando...");
     ESP.restart();
   }
 }
 
 // ─── MQTT ─────────────────────────────────────────────────────────────────────
 
 void connectMqtt() {
   Serial.print("[MQTT] Conectando a ");
   Serial.println(MQTT_BROKER);
 
   while (!mqttClient.connected()) {
     if (mqttClient.connect(MQTT_CLIENT)) {
       Serial.println("[MQTT] Conectado!");
       mqttClient.subscribe(TOPIC_ALERTS);
       Serial.print("[MQTT] Inscrito em: ");
       Serial.println(TOPIC_ALERTS);
     } else {
       Serial.print("[MQTT] Falha (rc=");
       Serial.print(mqttClient.state());
       Serial.println("). Tentando em 3s...");
       delay(3000);
     }
   }
 }
 
 // ─── Setup ────────────────────────────────────────────────────────────────────
 
 void setup() {
   Serial.begin(115200);
   delay(500);
 
   Serial.println("\n================================");
   Serial.println("  OrbitalShield — ESP32 Node");
   Serial.println("  Telemetria GNSS / Proxy Campo");
   Serial.println("================================\n");
 
   // LEDs
   pinMode(LED_GREEN,  OUTPUT);
   pinMode(LED_YELLOW, OUTPUT);
   pinMode(LED_RED,    OUTPUT);
   updateLeds();
 
   // Seed para ruído
   randomSeed(analogRead(0));
 
   // Conexões
   connectWifi();
 
   mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
   mqttClient.setCallback(onMqttMessage);
   mqttClient.setBufferSize(512);
   connectMqtt();
 
   Serial.println("\n[OK] Sistema iniciado. Aguardando alertas MQTT...\n");
 }
 
 // ─── Loop ─────────────────────────────────────────────────────────────────────
 
 void loop() {
   // Reconexão automática
   if (!mqttClient.connected()) {
     Serial.println("[MQTT] Desconectado. Reconectando...");
     connectMqtt();
   }
   mqttClient.loop();
 
   // Publica a cada PUBLISH_INTERVAL_MS
   unsigned long now = millis();
   if (now - lastPublish >= PUBLISH_INTERVAL_MS) {
     lastPublish = now;
     updateGnssState();
     publishTelemetry();
   }
 }
 
