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

- Base OMNIWeb de treino: **2018–2023**, com **52.553 linhas efetivas** após feature engineering e remoção de linhas inválidas
- Dados de 2024 reservados separadamente para validação, backtesting e replay
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
Dashboard Streamlit (calcula/visualiza OGII)
    ↓
risk_scores (SQLite) ← última inferência operacional no modo normal
    ↓
ingestion/mqtt_telemetry.py
    ↓
orbitalshield/alerts  →  ESP32 (orbital_shield.ino)
                                ↓
orbitalshield/esp32/telemetry  →  ingestion/mqtt_telemetry.py
                                ↓
                          esp32_telemetry (SQLite)
```

Fluxo complementar ARIMA:

```text
NASA/OMNIWeb → ARIMA (R) → kp_forecast.csv → Dashboard (tendência Kp 24h)
```

## Stack

- Python 3.11
- R 4.6 (ARIMA)
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
│   ├── ipo_definition.md
│   └── kp_arima_forecast.R
├── sprint0/
│   ├── 01_ipo_distribution.py
│   └── thresholds.json
├── validation/
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── setup.py
└── README.md
```

## Pré-requisitos

- Python 3.11+
- R 4.6+ (para projeção ARIMA de tendência Kp)
- SQLite (incluído no Python)

## Instalação

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Execução

### 1. Inicialização do banco

```bash
python -c "from db.connection import init_db; init_db()"
```

### 2. Ingestão de dados

```bash
python ingestion/omniweb_loader.py
python -c "from ingestion.omniweb_loader import load_historical; load_historical(2024, 2024)"
```

### 3. Sprint 0 — Gate científico

```bash
python sprint0/01_ipo_distribution.py
```

### 4. Treinamento

```bash
python model/train.py
```

> O arquivo `model/artifacts/xgboost_model.joblib` não é versionado no GitHub.
> Ele é gerado localmente por `python model/train.py`.

### 5. Backtesting

```bash
python backtesting/backtest_may2024.py
```

### 6. Projeção ARIMA — Tendência Kp 24h (R)

Execute da raiz do projeto:

```bash
Rscript research/kp_arima_forecast.R
```

O script instala/carrega os pacotes R necessários: `forecast`, `RSQLite`, `DBI`, `ggplot2` e `dplyr`.

### 7. Dashboard

```bash
streamlit run dashboard/app.py
```

> No modo normal, o dashboard atualiza a última inferência em `risk_scores`.
> Para a demo ESP32, abra o dashboard antes do bridge MQTT para popular `risk_scores` com um OGII real.

### Nota para demo ESP32/MQTT

Para que o ESP32 reflita o OGII atual, é necessário haver um registro recente em `risk_scores`.
Caso contrário, o bridge usa fallback MODERADO.

### 8. Bridge MQTT (ESP32 ↔ banco ↔ ESP32)

```bash
python ingestion/mqtt_telemetry.py
```

### 9. Firmware ESP32

Abra `esp32/orbital_shield.ino` na Arduino IDE.
Configure `WIFI_SSID` e `WIFI_PASSWORD` no sketch.
Para demonstração sem hardware físico: https://wokwi.com/projects/new/esp32

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
- Bridge Python lê o último OGII salvo em `risk_scores`, publica alertas MQTT e persiste telemetria em `esp32_telemetry` (SQLite)

## Tópicos MQTT

| Tópico | Direção | Payload |
|---|---|---|
| `orbitalshield/alerts` | `mqtt_telemetry.py` → ESP32 | `{ "ogii": 82, "level": "CRÍTICO" }` |
| `orbitalshield/esp32/telemetry` | ESP32 → `mqtt_telemetry.py` | `{ "hdop": 5.2, "satellites_visible": 5, ... }` |

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

- Variáveis sensíveis isoladas em `.env` — nunca versionado
- `.env.example` documenta variáveis sem expor valores reais
- IPO é constructo interno — não exposto na interface ou em logs
- OGII calculado exclusivamente em `model/predict.py`
- Credenciais do broker MQTT via `.env`
- Payload ESP32 com `is_replay: true` — transparência de dados simulados
- Test set maio/2024 usado uma única vez — resultados congelados
- Thresholds versionados em `sprint0/thresholds.json`

## Roadmap — Extensões Futuras

### Extensão 1 — Validação RBMC/IBGE

A Rede Brasileira de Monitoramento Contínuo GPS do IBGE registrou deriva de até **8,2 metros**
na estação CUIB (Cuiabá/MT) durante a tempestade de maio/2024. A próxima versão integra
dados RBMC/IBGE como ground truth real de degradação GNSS em solo brasileiro.

### Extensão 2 — OrbitalShield Rural (ConnectWindow)

Mais de **18 milhões de brasileiros** em regiões remotas dependem de satélites para GPS e
comunicação. A extensão ConnectWindow integra um agendador inteligente de comunicação
satelital: quando o OGII indica risco CRÍTICO e a janela de sinal é curta, o sistema prioriza
automaticamente mensagens de emergência médica e alertas de desastre.

| Público | Problema | Solução |
|---|---|---|
| Agricultor de precisão | GPS degrada sem aviso | OGII + 240h de antecipação |
| Agricultor familiar remoto | Não sabe quando o sinal chega | ConnectWindow — fila priorizada |
| Comunidades ribeirinhas | Emergências sem comunicação | Mensagens de saúde priorizadas |
| Gestores de desastre | Alertas não chegam a tempo | Alertas climáticos no topo da fila |

## Observações importantes

- Não versionar artefatos pesados ou arquivos sensíveis.
- Não expor o IPO na interface de usuário.
- Não recalibrar thresholds fora do processo formal do Sprint 0.
- Não tratar AMAS como causalidade comprovada.
- Payloads ESP32 com `is_replay: true` são dados de demonstração — não medição real de campo.

## Licença

Uso acadêmico interno, conforme regras do projeto e da FIAP.
