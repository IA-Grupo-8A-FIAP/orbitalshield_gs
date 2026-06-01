# OrbitalShield

Sistema de previsÃÂ£o de risco GNSS para agricultura de precisÃÂ£o com base em clima espacial.

## VisÃÂ£o geral

O **OrbitalShield** organiza um pipeline de quatro camadas para transformar dados de clima espacial em uma prediÃÂ§ÃÂ£o operacional de impacto sobre GNSS:

1. **IPO (constructo interno)**  
   ÃÂndice de PrevisÃÂ£o Operacional usado apenas na engenharia de atributos e no treinamento. O IPO **nÃÂ£o ÃÂ© exposto ao usuÃÂ¡rio final**.

2. **Modelo preditivo**  
   Classificador treinado com **XGBoost** para mapear o IPO em classes de risco.

3. **OGII (Operational GNSS Impact Index)**  
   ÃÂndice operacional calculado **apenas em `model/predict.py`**, em escala de **0 a 100**, para consumo externo.

4. **Telemetria de campo (ESP32)**  
   NÃÂ³ IoT que assina o alerta OGII via MQTT e simula degradaÃÂ§ÃÂ£o GNSS proporcional ao risco previsto Ã¢â¬â fechando o loop entre prediÃÂ§ÃÂ£o e impacto operacional.

## Regras cientÃÂ­ficas do projeto

- O IPO ÃÂ© um constructo interno e nÃÂ£o aparece na interface.
- O OGII ÃÂ© calculado somente no mÃÂ³dulo de inferÃÂªncia.
- O conjunto de teste de **maio/2024** foi usado **uma ÃÂºnica vez** no backtesting final.
- O modelo prevÃÂª risco em **t+1h** (horizonte de prediÃÂ§ÃÂ£o). O lead time operacional de 240h no evento de maio/2024 reflete detecÃÂ§ÃÂ£o contÃÂ­nua do inÃÂ­cio da rampa de degradaÃÂ§ÃÂ£o, nÃÂ£o previsÃÂ£o direta do pico.
- O AMAS ÃÂ© tratado como hipÃÂ³tese experimental; nÃÂ£o deve ser apresentado como causalidade.
- Os thresholds foram congelados apÃÂ³s o **Sprint 0** e nÃÂ£o devem ser recalibrados retroativamente.
- `is_replay: true` nos payloads do ESP32 indica dados simulados Ã¢â¬â nÃÂ£o confundir com mediÃÂ§ÃÂ£o real de campo.

## Resultados atuais

- Base OMNIWeb de treino: **2018Ã¢â¬â2023**, com **52.553 linhas efetivas** apÃÂ³s feature engineering e remoÃÂ§ÃÂ£o de linhas invÃÂ¡lidas
- Dados de 2024 reservados separadamente para validaÃÂ§ÃÂ£o, backtesting e replay
- Sprint 0 cientÃÂ­fico aprovado:
  - `p25 = 0.0305`
  - `p50 = 0.0592`
  - `p75 = 0.1053`
- Treinamento XGBoost:
  - **F1-macro = 0.8185**
  - **Recall classe 3 = 0.8729**
- Backtesting em evento de maio/2024:
  - **F1-macro = 0.8149**
  - **Recall classe 3 = 0.8919**
  - **Lead time operacional: 240 horas** Ã¢â¬â o modelo emitiu alertas CRÃÂTICO sequenciais hora a hora desde 01/05, detectando o inÃÂ­cio da rampa de degradaÃÂ§ÃÂ£o 10 dias antes do pico de Kp=9 em 11/05 (horizonte de prediÃÂ§ÃÂ£o: t+1h)

## Arquitetura

```text
Dados NOAA/OMNIWeb
    Ã¢â â
Ingestion + Feature Engineering
    Ã¢â â
IPO (interno)
    Ã¢â â
XGBoost
    Ã¢â â
OGII (operacional)
    Ã¢â â
Dashboard Streamlit (calcula/visualiza OGII)
    Ã¢â â
risk_scores (SQLite) Ã¢â Â ÃÂºltima inferÃÂªncia operacional no modo normal
    Ã¢â â
ingestion/mqtt_telemetry.py
    Ã¢â â
orbitalshield/alerts  Ã¢â â  ESP32 (orbital_shield.ino)
                                Ã¢â â
orbitalshield/esp32/telemetry  Ã¢â â  ingestion/mqtt_telemetry.py
                                Ã¢â â
                          esp32_telemetry (SQLite)
```

## Stack

- Python 3.11
- XGBoost
- Streamlit
- SQLite + SQLAlchemy
- MQTT (Paho)
- ESP32 + Arduino IDE

## Estrutura do repositÃÂ³rio

```text
orbitalshield_gs/
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ backtesting/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ backtest_may2024.py
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ results/
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ dashboard/
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ app.py
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ data/
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ reports/
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ db/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ connection.py
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ models.py
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ esp32/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ orbital_shield.ino
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ README.md
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ experiments/
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ features/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ engineering.py
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ ipo.py
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ ingestion/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ omniweb_loader.py
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ noaa_collector.py
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ mqtt_telemetry.py
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ model/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ artifacts/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ train.py
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ predict.py
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ research/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ ipo_definition.md
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ kp_arima_forecast.R
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ sprint0/
Ã¢ââ   Ã¢âÅÃ¢ââ¬Ã¢ââ¬ 01_ipo_distribution.py
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ thresholds.json
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ validation/
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ .env.example
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ .gitignore
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ .streamlit/
Ã¢ââ   Ã¢ââÃ¢ââ¬Ã¢ââ¬ config.toml
Ã¢âÅÃ¢ââ¬Ã¢ââ¬ setup.py
Ã¢ââÃ¢ââ¬Ã¢ââ¬ README.md
```

## PrÃÂ©-requisitos

- Python 3.11+
- R 4.6+ para a projeÃÂ§ÃÂ£o ARIMA de tendÃÂªncia Kp
- SQLite via biblioteca padrÃÂ£o do Python

## InstalaÃÂ§ÃÂ£o

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
.venv\Scripts\activate.bat
```

Linux/macOS:
```bash
source .venv/bin/activate
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## ExecuÃÂ§ÃÂ£o

### 1. InicializaÃÂ§ÃÂ£o do banco
```bash
python -c "from db.connection import init_db; init_db()"
```

### 2. IngestÃÂ£o de dados
```bash
python ingestion/omniweb_loader.py
python -c "from ingestion.omniweb_loader import load_historical; load_historical(2024, 2024)"
```

### 3. Sprint 0 Ã¢â¬â Gate cientÃÂ­fico
```bash
python sprint0/01_ipo_distribution.py
```

### 4. Treinamento
```bash
python model/train.py
```

> O arquivo `model/artifacts/xgboost_model.joblib` nÃÂ£o ÃÂ© versionado no GitHub.
> Ele ÃÂ© gerado localmente por `python model/train.py`.

### 5. Backtesting
```bash
python backtesting/backtest_may2024.py
```

### 6. ProjeÃÂ§ÃÂ£o ARIMA Ã¢â¬â TendÃÂªncia Kp 24h (R)
Execute da raiz do projeto:
```bash
Rscript research/kp_arima_forecast.R
```

O script instala/carrega os pacotes R necessÃÂ¡rios: `forecast`, `RSQLite`, `DBI`, `ggplot2` e `dplyr`.

### 7. Dashboard
```bash
streamlit run dashboard/app.py
```

> No modo normal, o dashboard atualiza a ÃÂºltima inferÃÂªncia em `risk_scores`.
> Caso a tabela esteja vazia, o bridge MQTT usa fallback `MODERADO`.
> Para a demo ESP32, abra o dashboard em modo normal antes da bridge MQTT para popular `risk_scores` com um OGII real.

### 8. Bridge MQTT (ESP32 Ã¢â â banco Ã¢â â ESP32)
```bash
python ingestion/mqtt_telemetry.py
```

### 9. Firmware ESP32
Abra `esp32/orbital_shield.ino` na Arduino IDE.  
Configure `WIFI_SSID` e `WIFI_PASSWORD` no sketch.  
Para demonstraÃÂ§ÃÂ£o sem hardware fÃÂ­sico: [Wokwi](https://wokwi.com/projects/new/esp32)

### ObservaÃÂ§ÃÂ£o sobre artefatos locais

O banco `orbitalshield.db` e o modelo `xgboost_model.joblib` nÃÂ£o sÃÂ£o versionados.
Eles sÃÂ£o recriados pelos passos de inicializaÃÂ§ÃÂ£o, ingestÃÂ£o e treinamento para manter
rastreabilidade e evitar versionar arquivos pesados ou sensÃÂ­veis.

## OrganizaÃÂ§ÃÂ£o por camadas

### Camada 1 Ã¢â¬â IPO
- DefiniÃÂ§ÃÂ£o interna do ÃÂ­ndice
- Feature engineering orientado por fÃÂ­sica de clima espacial
- Thresholds congelados apÃÂ³s Sprint 0

### Camada 2 Ã¢â¬â Modelo
- Treinamento com XGBoost
- PersistÃÂªncia de artefatos em `model/artifacts/`
- Metadados de modelo e thresholds versionados

### Camada 3 Ã¢â¬â OGII
- ConversÃÂ£o da saÃÂ­da do modelo para ÃÂ­ndice operacional 0Ã¢â¬â100
- ExposiÃÂ§ÃÂ£o para dashboard, telemetria e integraÃÂ§ÃÂµes

### Camada 4 Ã¢â¬â Telemetria ESP32
- NÃÂ³ IoT que assina `orbitalshield/alerts` via MQTT
- Simula degradaÃÂ§ÃÂ£o GNSS (HDOP, satÃÂ©lites, fix) proporcional ao OGII
- Publica `orbitalshield/esp32/telemetry` a cada 5s
- Bridge Python lÃÂª o ÃÂºltimo OGII salvo em `risk_scores`, publica alertas MQTT e persiste telemetria em `esp32_telemetry` (SQLite)

## TÃÂ³picos MQTT

| TÃÂ³pico | DireÃÂ§ÃÂ£o | Payload |
|---|---|---|
| `orbitalshield/alerts` | `mqtt_telemetry.py` Ã¢â â ESP32 | `{ "ogii": 82, "level": "CRÃÂTICO" }` |
| `orbitalshield/esp32/telemetry` | ESP32 Ã¢â â `mqtt_telemetry.py` | `{ "hdop": 5.2, "satellites_visible": 5, ... }` |

## ValidaÃÂ§ÃÂ£o em trÃÂªs camadas

| Camada | O que valida | Resultado |
|---|---|---|
| EstatÃÂ­stica | F1-macro, recall crÃÂ­tico no test set | 0.8149 / 0.8919 |
| Operacional | OGII + recomendaÃÂ§ÃÂ£o RTK no dashboard | AntecipaÃÂ§ÃÂ£o de 240h |
| Proxy fÃÂ­sico | HDOP e satÃÂ©lites via ESP32 | CorrelaÃÂ§ÃÂ£o com alert_level |

## Integrantes do grupo

| Nome | E-mail | RM |
|---|---|---|
| Lucas Carvalho Cordeiro | carvalho.lucascc@gmail.com | 570388 |
| Larissa da Silva Marcelino | larissamarcelinocpb@gmail.com | 571790 |
| Abner Henrique Dias Rosa Sanches | abner.mtpvp@gmail.com | 572253 |
| Brenoezo Leardini | b.leardini@gmail.com | 572533 |
| Elton Modesto de Souza Dias | elton.redes@hotmail.com | 572530 |


## SeguranÃÂ§a

O projeto implementa prÃÂ¡ticas de seguranÃÂ§a em mÃÂºltiplas camadas:

### ProteÃÂ§ÃÂ£o de credenciais
- VariÃÂ¡veis sensÃÂ­veis (broker MQTT, caminhos, chaves) isoladas em `.env`
- `.env` protegido pelo `.gitignore` Ã¢â¬â nunca versionado
- `.env.example` documenta as variÃÂ¡veis sem expor valores reais
- ValidaÃÂ§ÃÂ£o de variÃÂ¡veis obrigatÃÂ³rias no startup via `db/connection.py`

### SeparaÃÂ§ÃÂ£o de camadas
- IPO ÃÂ© constructo interno Ã¢â¬â nÃÂ£o exposto na interface ou em logs
- OGII calculado exclusivamente em `model/predict.py`
- Artefatos do modelo (`.joblib`) no `.gitignore` Ã¢â¬â nÃÂ£o versionados

### IoT / MQTT
- Credenciais do broker via `.env` (nunca hardcoded em produÃÂ§ÃÂ£o)
- Payload ESP32 com `is_replay: true` Ã¢â¬â transparÃÂªncia de dados simulados
- TÃÂ³picos com namespace dedicado (`orbitalshield/`)

### Dados e rastreabilidade
- Test set maio/2024 usado uma ÃÂºnica vez Ã¢â¬â resultados congelados
- Thresholds versionados em `sprint0/thresholds.json`
- Banco SQLite local Ã¢â¬â dados nÃÂ£o expostos a serviÃÂ§os externos

### PrÃÂ³ximos passos de seguranÃÂ§a (fase 2)
- TLS no broker MQTT (porta 8883)
- AutenticaÃÂ§ÃÂ£o username/password no broker
- Rate limiting no dashboard para deploy pÃÂºblico

## Roadmap Ã¢â¬â ExtensÃÂµes Futuras

O OrbitalShield foi concebido para agricultura de precisÃÂ£o, mas o problema do clima espacial ÃÂ© mais amplo. O grupo identificou duas extensÃÂµes naturais do sistema:

### ExtensÃÂ£o 1 Ã¢â¬â ValidaÃÂ§ÃÂ£o RBMC/IBGE (SunStrike)

A Rede Brasileira de Monitoramento ContÃÂ­nuo GPS do IBGE registrou deriva de posicionamento de atÃÂ© **8,2 metros** na estaÃÂ§ÃÂ£o CUIB (CuiabÃÂ¡/MT) durante a tempestade de maio/2024. Esses dados RINEX sÃÂ£o pÃÂºblicos e representam o **ground truth real** de degradaÃÂ§ÃÂ£o GNSS em solo brasileiro.

A prÃÂ³xima versÃÂ£o do OrbitalShield integrarÃÂ¡ os dados RBMC/IBGE como validaÃÂ§ÃÂ£o fÃÂ­sica direta Ã¢â¬â substituindo o proxy ESP32 por mediÃÂ§ÃÂµes reais de receptor geodÃÂ©sico, eliminando a principal limitaÃÂ§ÃÂ£o cientÃÂ­fica atual.

### ExtensÃÂ£o 2 Ã¢â¬â OrbitalShield Rural (ConnectWindow)

Mais de **18 milhÃÂµes de brasileiros** em regiÃÂµes remotas (comunidades ribeirinhas, quilombolas, agricultores familiares) dependem de satÃÂ©lites tanto para GPS quanto para comunicaÃÂ§ÃÂ£o. Nessas regiÃÂµes, a janela de sinal satelital ÃÂ© intermitente Ã¢â¬â e ninguÃÂ©m avisa quando ela chega.

A extensÃÂ£o ConnectWindow integra ao OrbitalShield:

- **Simulador orbital** Ã¢â¬â calcula quando um satÃÂ©lite passa sobre uma coordenada usando dados TLE do Celestrak/NASA
- **Fila inteligente** Ã¢â¬â prioriza mensagens por urgÃÂªncia (emergÃÂªncia mÃÂ©dica Ã¢â â alerta climÃÂ¡tico Ã¢â â dados agrÃÂ­colas Ã¢â â comunicaÃÂ§ÃÂ£o pessoal)
- **Otimizador de janela** Ã¢â¬â dado X minutos de sinal e banda limitada, decide o que enviar primeiro
- **Preditor com ML** Ã¢â¬â regressÃÂ£o linear treinada com histÃÂ³rico de janelas para prever duraÃÂ§ÃÂ£o e qualidade do prÃÂ³ximo sinal

**IntegraÃÂ§ÃÂ£o com o OGII:** quando o ÃÂ­ndice indica risco CRÃÂTICO E a janela de comunicaÃÂ§ÃÂ£o ÃÂ© curta, o sistema prioriza automaticamente mensagens de emergÃÂªncia mÃÂ©dica e alertas de desastre Ã¢â¬â tecnologia espacial como ferramenta de inclusÃÂ£o digital.

### Impacto social ampliado

| PÃÂºblico | Problema | SoluÃÂ§ÃÂ£o |
|---|---|---|
| Agricultor de precisÃÂ£o | GPS degrada sem aviso | OGII + alerta 240h de antecipaÃÂ§ÃÂ£o |
| Agricultor familiar remoto | NÃÂ£o sabe quando o sinal chega | ConnectWindow Ã¢â¬â fila priorizada |
| Comunidades ribeirinhas | EmergÃÂªncias mÃÂ©dicas sem comunicaÃÂ§ÃÂ£o | Mensagens de saÃÂºde priorizadas na janela |
| Gestores de desastre | Alertas de enchente nÃÂ£o chegam | Alertas climÃÂ¡ticos no topo da fila |

## ObservaÃÂ§ÃÂµes importantes

- NÃÂ£o versionar artefatos pesados ou arquivos sensÃÂ­veis.
- NÃÂ£o expor o IPO na interface de usuÃÂ¡rio.
- NÃÂ£o recalibrar thresholds fora do processo formal do Sprint 0.
- NÃÂ£o tratar AMAS como causalidade comprovada.
- Payloads ESP32 com `is_replay: true` sÃÂ£o dados de demonstraÃÂ§ÃÂ£o Ã¢â¬â nÃÂ£o mediÃÂ§ÃÂ£o real de campo.

## LicenÃÂ§a

Uso acadÃÂªmico interno, conforme regras do projeto e da FIAP.

