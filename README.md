# OrbitalShield

Sistema de previsao de risco GNSS para agricultura de precisao com base em clima espacial.

## Visao geral

O **OrbitalShield** organiza um pipeline de quatro camadas para transformar dados de clima espacial em uma predicao operacional de impacto sobre GNSS:

1. **IPO (constructo interno)**
   Indice de Previsao Operacional usado apenas na engenharia de atributos e no treinamento. O IPO **nao e exposto ao usuario final**.

2. **Modelo preditivo**
   Classificador treinado com **XGBoost** para mapear o IPO em classes de risco.

3. **OGII (Operational GNSS Impact Index)**
   Indice operacional calculado **apenas em `model/predict.py`**, em escala de **0 a 100**, para consumo externo.

4. **Telemetria de campo (ESP32)**
   No IoT que assina o alerta OGII via MQTT e simula degradacao GNSS proporcional ao risco previsto -- fechando o loop entre predicao e impacto operacional.

## Regras cientificas do projeto

- O IPO e um constructo interno e nao aparece na interface.
- O OGII e calculado somente no modulo de inferencia.
- O conjunto de teste de **maio/2024** foi usado **uma unica vez** no backtesting final.
- O modelo preve risco em **t+1h** (horizonte de predicao). O lead time operacional de 240h no evento de maio/2024 reflete deteccao continua do inicio da rampa de degradacao, nao previsao direta do pico.
- O AMAS e tratado como hipotese experimental; nao deve ser apresentado como causalidade.
- Os thresholds foram congelados apos o **Sprint 0** e nao devem ser recalibrados retroativamente.
- `is_replay: true` nos payloads do ESP32 indica dados simulados -- nao confundir com medicao real de campo.

## Resultados atuais

- Base OMNIWeb de treino: **2018-2023**, com **52.553 linhas efetivas** apos feature engineering e remocao de linhas invalidas
- Dados de 2024 reservados separadamente para validacao, backtesting e replay
- Sprint 0 cientifico aprovado:
  - `p25 = 0.0305`
  - `p50 = 0.0592`
  - `p75 = 0.1053`
- Treinamento XGBoost:
  - **F1-macro = 0.8185**
  - **Recall classe 3 = 0.8729**
- Backtesting em evento de maio/2024:
  - **F1-macro = 0.8149**
  - **Recall classe 3 = 0.8919**
  - **Lead time operacional: 240 horas** -- o modelo emitiu alertas CRITICO sequenciais hora a hora desde 01/05, detectando o inicio da rampa de degradacao 10 dias antes do pico de Kp=9 em 11/05 (horizonte de predicao: t+1h)

## Arquitetura

```
Dados NOAA/OMNIWeb
    |
Ingestion + Feature Engineering
    |
IPO (interno)
    |
XGBoost
    |
OGII (operacional)
    |
Dashboard Streamlit (calcula/visualiza OGII)
    |
risk_scores (SQLite) <- ultima inferencia operacional no modo normal
    |
ingestion/mqtt_telemetry.py
    |
orbitalshield/alerts  ->  ESP32 (orbital_shield.ino)
                                |
orbitalshield/esp32/telemetry  ->  ingestion/mqtt_telemetry.py
                                |
                          esp32_telemetry (SQLite)
```

Fluxo complementar ARIMA:
```
NASA/OMNIWeb -> ARIMA (R) -> kp_forecast.csv -> Dashboard (tendencia Kp 24h)
```

## Stack

- Python 3.11
- R 4.6 (ARIMA)
- XGBoost
- Streamlit
- SQLite + SQLAlchemy
- MQTT (Paho)
- ESP32 + Arduino IDE

## Estrutura do repositorio

```
orbitalshield_gs/
|-- backtesting/
|   |-- backtest_may2024.py
|   `-- results/
|-- dashboard/
|   `-- app.py
|-- data/
|   `-- reports/
|-- db/
|   |-- connection.py
|   `-- models.py
|-- esp32/
|   |-- orbital_shield.ino
|   `-- README.md
|-- experiments/
|-- features/
|   |-- engineering.py
|   `-- ipo.py
|-- ingestion/
|   |-- omniweb_loader.py
|   |-- noaa_collector.py
|   `-- mqtt_telemetry.py
|-- model/
|   |-- artifacts/
|   |-- train.py
|   `-- predict.py
|-- research/
|   |-- ipo_definition.md
|   `-- kp_arima_forecast.R
|-- sprint0/
|   |-- 01_ipo_distribution.py
|   `-- thresholds.json
|-- validation/
|-- .env.example
|-- .gitignore
|-- .streamlit/
|   `-- config.toml
|-- requirements.txt
|-- setup.py
`-- README.md
```

## Pre-requisitos

- Python 3.11+
- R 4.6+ (para projecao ARIMA de tendencia Kp)
- SQLite (incluido no Python)

## Instalacao

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

## Execucao

### 1. Inicializacao do banco
```bash
python -c "from db.connection import init_db; init_db()"
```

### 2. Ingestao de dados
```bash
python ingestion/omniweb_loader.py
python -c "from ingestion.omniweb_loader import load_historical; load_historical(2024, 2024)"
```

### 3. Sprint 0 -- Gate cientifico
```bash
python sprint0/01_ipo_distribution.py
```

### 4. Treinamento
```bash
python model/train.py
```
> O arquivo `model/artifacts/xgboost_model.joblib` nao e versionado no GitHub.
> Ele e gerado localmente por `python model/train.py`.

### 5. Backtesting
```bash
python backtesting/backtest_may2024.py
```

### 6. Projecao ARIMA -- Tendencia Kp 24h (R)
Execute da raiz do projeto:
```bash
Rscript research/kp_arima_forecast.R
```
O script instala/carrega os pacotes R necessarios: `forecast`, `RSQLite`, `DBI`, `ggplot2` e `dplyr`.

### 7. Dashboard
```bash
streamlit run dashboard/app.py
```
> No modo normal, o dashboard atualiza a ultima inferencia em `risk_scores`.
> Para a demo ESP32, abra o dashboard antes do bridge MQTT para popular `risk_scores` com um OGII real.

### Nota para demo ESP32/MQTT
Para que o ESP32 reflita o OGII atual, e necessario haver um registro recente em `risk_scores`.
Caso contrario, o bridge usa fallback MODERADO.

### 8. Bridge MQTT (ESP32 <-> banco <-> ESP32)
```bash
python ingestion/mqtt_telemetry.py
```

### 9. Firmware ESP32
Abra `esp32/orbital_shield.ino` na Arduino IDE.
Configure `WIFI_SSID` e `WIFI_PASSWORD` no sketch.
Para demonstracao sem hardware fisico: https://wokwi.com/projects/new/esp32

## Organizacao por camadas

### Camada 1 -- IPO
- Definicao interna do indice
- Feature engineering orientado por fisica de clima espacial
- Thresholds congelados apos Sprint 0

### Camada 2 -- Modelo
- Treinamento com XGBoost
- Persistencia de artefatos em `model/artifacts/`
- Metadados de modelo e thresholds versionados

### Camada 3 -- OGII
- Conversao da saida do modelo para indice operacional 0-100
- Exposicao para dashboard, telemetria e integracoes

### Camada 4 -- Telemetria ESP32
- No IoT que assina `orbitalshield/alerts` via MQTT
- Simula degradacao GNSS (HDOP, satelites, fix) proporcional ao OGII
- Publica `orbitalshield/esp32/telemetry` a cada 5s
- Bridge Python le o ultimo OGII salvo em `risk_scores`, publica alertas MQTT e persiste telemetria em `esp32_telemetry` (SQLite)

## Topicos MQTT

| Topico | Direcao | Payload |
|---|---|---|
| `orbitalshield/alerts` | `mqtt_telemetry.py` -> ESP32 | `{ "ogii": 82, "level": "CRITICO" }` |
| `orbitalshield/esp32/telemetry` | ESP32 -> `mqtt_telemetry.py` | `{ "hdop": 5.2, "satellites_visible": 5, ... }` |

## Validacao em tres camadas

| Camada | O que valida | Resultado |
|---|---|---|
| Estatistica | F1-macro, recall critico no test set | 0.8149 / 0.8919 |
| Operacional | OGII + recomendacao RTK no dashboard | Antecipacao de 240h |
| Proxy fisico | HDOP e satelites via ESP32 | Correlacao com alert_level |

## Integrantes do grupo

| Nome | E-mail | RM |
|---|---|---|
| Lucas Carvalho Cordeiro | carvalho.lucascc@gmail.com | 570388 |
| Larissa da Silva Marcelino | larissamarcelinocpb@gmail.com | 571790 |
| Abner Henrique Dias Rosa Sanches | abner.mtpvp@gmail.com | 572253 |
| Brenoezo Leardini | b.leardini@gmail.com | 572533 |
| Elton Modesto de Souza Dias | elton.redes@hotmail.com | 572530 |

## Seguranca

- Variaveis sensiveis isoladas em `.env` -- nunca versionado
- `.env.example` documenta variaveis sem expor valores reais
- IPO e constructo interno -- nao exposto na interface ou em logs
- OGII calculado exclusivamente em `model/predict.py`
- Credenciais do broker MQTT via `.env`
- Payload ESP32 com `is_replay: true` -- transparencia de dados simulados
- Test set maio/2024 usado uma unica vez -- resultados congelados
- Thresholds versionados em `sprint0/thresholds.json`

## Roadmap -- Extensoes Futuras

### Extensao 1 -- Validacao RBMC/IBGE
A Rede Brasileira de Monitoramento Continuo GPS do IBGE registrou deriva de ate **8,2 metros**
na estacao CUIB (Cuiaba/MT) durante a tempestade de maio/2024. A proxima versao integra
dados RBMC/IBGE como ground truth real de degradacao GNSS em solo brasileiro.

### Extensao 2 -- OrbitalShield Rural (ConnectWindow)
Mais de **18 milhoes de brasileiros** em regioes remotas dependem de satelites para GPS e
comunicacao. A extensao ConnectWindow integra um agendador inteligente de comunicacao
satelital: quando o OGII indica risco CRITICO e a janela de sinal e curta, o sistema prioriza
automaticamente mensagens de emergencia medica e alertas de desastre.

| Publico | Problema | Solucao |
|---|---|---|
| Agricultor de precisao | GPS degrada sem aviso | OGII + 240h de antecipacao |
| Agricultor familiar remoto | Nao sabe quando o sinal chega | ConnectWindow -- fila priorizada |
| Comunidades ribeirinhas | Emergencias sem comunicacao | Mensagens de saude priorizadas |
| Gestores de desastre | Alertas nao chegam a tempo | Alertas climaticos no topo da fila |

## Observacoes importantes

- Nao versionar artefatos pesados ou arquivos sensiveis.
- Nao expor o IPO na interface de usuario.
- Nao recalibrar thresholds fora do processo formal do Sprint 0.
- Nao tratar AMAS como causalidade comprovada.
- Payloads ESP32 com `is_replay: true` sao dados de demonstracao -- nao medicao real de campo.

## Licenca

Uso academico interno, conforme regras do projeto e da FIAP.