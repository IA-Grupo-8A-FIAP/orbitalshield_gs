# OrbitalShield

Sistema de previsão de risco GNSS para agricultura de precisão com base em clima espacial.

## Visão geral

O **OrbitalShield** organiza um pipeline de quatro camadas para transformar dados de clima espacial em uma predição operacional de impacto sobre GNSS:

1. **IPO (constructo interno)**  
   Índice de Previsão Operacional usado apenas na engenharia de atributos e no treinamento. O IPO **não é exposto ao usuário final**.

2. **Modelo preditivo**  
   Classificador treinado com **XGBoost** para mapear o IPO em classes de risco.

3. **OGII (Operational GNSS Impact Index)**  
   Índice operacional calculado **apenas em `model/predict.py`**, em escala de **0 a 100**, para consumo externo.

4. **Telemetria de campo (ESP32)**  
   Nó IoT que assina o alerta OGII via MQTT e simula degradação GNSS proporcional ao risco previsto — fechando o loop entre predição e impacto operacional.

## Regras científicas do projeto

- O IPO é um constructo interno e não aparece na interface.
- O OGII é calculado somente no módulo de inferência.
- O conjunto de teste de **maio/2024** foi usado **uma única vez** no backtesting final.
- O modelo prevê risco em **t+1h** (horizonte de predição). O lead time operacional de 240h no evento de maio/2024 reflete detecção contínua do início da rampa de degradação, não previsão direta do pico.
- O AMAS é tratado como hipótese experimental; não deve ser apresentado como causalidade.
- Os thresholds foram congelados após o **Sprint 0** e não devem ser recalibrados retroativamente.
- `is_replay: true` nos payloads do ESP32 indica dados simulados — não confundir com medição real de campo.

## Resultados atuais

- Ingestão histórica NOAA/OMNIWeb: **52.584 registros** (2018–2024)
- Sprint 0 científico aprovado:
  - `p25 = 0.0305`
  - `p50 = 0.0592`
  - `p75 = 0.1053`
- Treinamento XGBoost:
  - **F1-macro = 0.8185**
  - **Recall classe 3 = 0.8729**
- Backtesting em evento de maio/2024:
  - **F1-macro = 0.8149**
  - **Recall classe 3 = 0.8919**
  - **Lead time operacional: 240 horas** — o modelo emitiu alertas CRÍTICO sequenciais hora a hora desde 01/05, detectando o início da rampa de degradação 10 dias antes do pico de Kp=9 em 11/05 (horizonte de predição: t+1h)

## Arquitetura

```text
Dados NOAA/OMNIWeb
    ↓
Ingestion + Feature Engineering
    ↓
IPO (interno)
    ↓
XGBoost
    ↓
OGII (operacional)
    ↓
Dashboard Streamlit
    ↓
orbitalshield/alerts  →  ESP32 (orbital_shield.ino)
                                ↓
orbitalshield/esp32/telemetry  →  ingestion/mqtt_telemetry.py
                                ↓
                          esp32_telemetry (SQLite)
```

## Stack

- Python 3.11
- XGBoost
- Streamlit
- SQLite + SQLAlchemy
- MQTT (Paho)
- ESP32 + Arduino IDE

## Estrutura do repositório

```text
orbitalshield_gs/
├── backtesting/
│   ├── backtest_may2024.py
│   └── results/
├── dashboard/
│   └── app.py
├── data/
│   └── reports/
├── db/
│   ├── connection.py
│   └── models.py
├── esp32/
│   ├── orbital_shield.ino
│   └── README.md
├── experiments/
├── features/
│   ├── engineering.py
│   └── ipo.py
├── ingestion/
│   ├── omniweb_loader.py
│   ├── noaa_collector.py
│   └── mqtt_telemetry.py
├── model/
│   ├── artifacts/
│   ├── train.py
│   └── predict.py
├── research/
│   └── ipo_definition.md
├── sprint0/
│   ├── 01_ipo_distribution.py
│   └── thresholds.json
├── validation/
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── setup.py
└── README.md
```

## Instalação

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

## Execução

### 1. Ingestão de dados
```bash
python ingestion/omniweb_loader.py
```

### 2. Sprint 0 — Gate científico
```bash
python sprint0/01_ipo_distribution.py
```

### 3. Treinamento
```bash
python model/train.py
```

### 4. Backtesting
```bash
python backtesting/backtest_may2024.py
```

### 5. Dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Bridge MQTT (ESP32 ↔ Dashboard)
```bash
python ingestion/mqtt_telemetry.py
```

### 7. Firmware ESP32
Abra `esp32/orbital_shield.ino` na Arduino IDE.  
Configure `WIFI_SSID` e `WIFI_PASSWORD` no sketch.  
Para demonstração sem hardware físico: [Wokwi](https://wokwi.com/projects/new/esp32)

## Organização por camadas

### Camada 1 — IPO
- Definição interna do índice
- Feature engineering orientado por física de clima espacial
- Thresholds congelados após Sprint 0

### Camada 2 — Modelo
- Treinamento com XGBoost
- Persistência de artefatos em `model/artifacts/`
- Metadados de modelo e thresholds versionados

### Camada 3 — OGII
- Conversão da saída do modelo para índice operacional 0–100
- Exposição para dashboard, telemetria e integrações

### Camada 4 — Telemetria ESP32
- Nó IoT que assina `orbitalshield/alerts` via MQTT
- Simula degradação GNSS (HDOP, satélites, fix) proporcional ao OGII
- Publica `orbitalshield/esp32/telemetry` a cada 5s
- Bridge Python persiste dados em `esp32_telemetry` (SQLite)

## Tópicos MQTT

| Tópico | Direção | Payload |
|---|---|---|
| `orbitalshield/alerts` | Dashboard → ESP32 | `{ "ogii": 82, "level": "CRÍTICO" }` |
| `orbitalshield/esp32/telemetry` | ESP32 → Dashboard | `{ "hdop": 5.2, "satellites_visible": 5, ... }` |

## Validação em três camadas

| Camada | O que valida | Resultado |
|---|---|---|
| Estatística | F1-macro, recall crítico no test set | 0.8149 / 0.8919 |
| Operacional | OGII + recomendação RTK no dashboard | Antecipação de 240h |
| Proxy físico | HDOP e satélites via ESP32 | Correlação com alert_level |

## Integrantes do grupo

| Nome | E-mail | RM |
|---|---|---|
| Lucas Carvalho Cordeiro | carvalho.lucascc@gmail.com | 570388 |
| Larissa da Silva Marcelino | larissamarcelinocpb@gmail.com | 571790 |
| Abner Henrique Dias Rosa Sanches | abner.mtpvp@gmail.com | 572253 |
| Brenoezo Leardini | b.leardini@gmail.com | 572533 |
| Elton Modesto de Souza Dias | elton.redes@hotmail.com | 572530 |


## Segurança

O projeto implementa práticas de segurança em múltiplas camadas:

### Proteção de credenciais
- Variáveis sensíveis (broker MQTT, caminhos, chaves) isoladas em `.env`
- `.env` protegido pelo `.gitignore` — nunca versionado
- `.env.example` documenta as variáveis sem expor valores reais
- Validação de variáveis obrigatórias no startup via `db/connection.py`

### Separação de camadas
- IPO é constructo interno — não exposto na interface ou em logs
- OGII calculado exclusivamente em `model/predict.py`
- Artefatos do modelo (`.joblib`) no `.gitignore` — não versionados

### IoT / MQTT
- Credenciais do broker via `.env` (nunca hardcoded em produção)
- Payload ESP32 com `is_replay: true` — transparência de dados simulados
- Tópicos com namespace dedicado (`orbitalshield/`)

### Dados e rastreabilidade
- Test set maio/2024 usado uma única vez — resultados congelados
- Thresholds versionados em `sprint0/thresholds.json`
- Banco SQLite local — dados não expostos a serviços externos

### Próximos passos de segurança (fase 2)
- TLS no broker MQTT (porta 8883)
- Autenticação username/password no broker
- Rate limiting no dashboard para deploy público

## Observações importantes

- Não versionar artefatos pesados ou arquivos sensíveis.
- Não expor o IPO na interface de usuário.
- Não recalibrar thresholds fora do processo formal do Sprint 0.
- Não tratar AMAS como causalidade comprovada.
- Payloads ESP32 com `is_replay: true` são dados de demonstração — não medição real de campo.

## Licença

Uso acadêmico interno, conforme regras do projeto e da FIAP.