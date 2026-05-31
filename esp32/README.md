# ESP32 — OrbitalShield Node

Nó de telemetria GNSS para validação física indireta do OGII.

## Hipótese científica

Tempestades geomagnéticas degradam o sinal GNSS: o HDOP sobe e o número de satélites visíveis cai. O ESP32 representa um receptor GNSS em campo (trator ou drone agrícola) que reage às previsões do modelo — fechando o loop entre predição e impacto operacional.

## Dependências (Arduino Library Manager)

- PubSubClient by Nick O'Leary >= 2.8
- ArduinoJson by Benoit Blanchon >= 6.21

## Configuração

Edite orbital_shield.ino e substitua:

    const char* WIFI_SSID     = "SEU_WIFI";
    const char* WIFI_PASSWORD = "SUA_SENHA";

## Tópicos MQTT

| Tópico | Direção | Payload |
|--------|---------|---------|
| orbitalshield/alerts | Subscribe | { "ogii": 82, "level": "CRÍTICO" } |
| orbitalshield/esp32/telemetry | Publish | { "hdop": 5.2, "satellites_visible": 5, ... } |

## Máquina de estados

| Nível | HDOP | Satélites | Status |
|-------|------|-----------|--------|
| BAIXO | 0.8–1.4 | 12–16 | OK |
| MODERADO | 1.5–2.5 | 9–12 | DEGRADED |
| ALTO | 2.5–4.0 | 6–9 | DEGRADED |
| CRÍTICO | 4.0–8.0 | 3–6 | NO_RTK |

## Simulação no Wokwi

1. Acesse https://wokwi.com/projects/new/esp32
2. Cole o conteúdo de orbital_shield.ino
3. Adicione LEDs nos pinos 2, 4 e 5
4. Rode python ingestion/mqtt_telemetry.py para fechar o loop

## Observações

- is_replay: true indica dados simulados
- O ESP32 não prevê clima espacial — representa o impacto operacional previsto pelo modelo
