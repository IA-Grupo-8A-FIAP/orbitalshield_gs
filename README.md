# OrbitalShield

Sistema de previsÃ£o de risco GNSS para agricultura de precisÃ£o com base em clima espacial.

## VisÃ£o geral

O **OrbitalShield** organiza um pipeline de quatro camadas para transformar dados de clima espacial em uma prediÃ§Ã£o operacional de impacto sobre GNSS:

1. **IPO (constructo interno)**  
   Ãndice de PrevisÃ£o Operacional usado apenas na engenharia de atributos e no treinamento. O IPO **nÃ£o Ã© exposto ao usuÃ¡rio final**.

2. **Modelo preditivo**  
   Classificador treinado com **XGBoost** para mapear o IPO em classes de risco.

3. **OGII (Operational GNSS Impact Index)**  
   Ãndice operacional calculado **apenas em `model/predict.py`**, em escala de **0 a 100**, para consumo externo.

4. **Telemetria de campo (ESP32)**  
   NÃ³ IoT que assina o alerta OGII via MQTT e simula degradaÃ§Ã£o GNSS proporcional ao risco previsto â€” fechando o loop entre prediÃ§Ã£o e impacto operacional.

## Regras cientÃ­ficas do projeto

- O IPO Ã© um constructo interno e nÃ£o aparece na interface.
- O OGII Ã© calculado somente no mÃ³dulo de inferÃªncia.
- O conjunto de teste de **maio/2024** foi usado **uma Ãºnica vez** no backtesting final.
- O modelo prevÃª risco em **t+1h** (horizonte de prediÃ§Ã£o). O lead time operacional de 240h no evento de maio/2024 reflete detecÃ§Ã£o contÃ­nua do inÃ­cio da rampa de degradaÃ§Ã£o, nÃ£o previsÃ£o direta do pico.
- O AMAS Ã© tratado como hipÃ³tese experimental; nÃ£o deve ser apresentado como causalidade.
- Os thresholds foram congelados apÃ³s o **Sprint 0** e nÃ£o devem ser recalibrados retroativamente.
- `is_replay: true` nos payloads do ESP32 indica dados simulados â€” nÃ£o confundir com mediÃ§Ã£o real de campo.

## Resultados atuais

- Base OMNIWeb de treino: **2018â€“2023**, com **52.553 linhas efetivas** apÃ³s feature engineering e remoÃ§Ã£o de linhas invÃ¡lidas
- Dados de 2024 reservados separadamente para validaÃ§Ã£o, backtesting e replay
- Sprint 0 cientÃ­fico aprovado:
  - `p25 = 0.0305`
  - `p50 = 0.0592`
  - `p75 = 0.1053`
- Treinamento XGBoost:
  - **F1-macro = 0.8185**
  - **Recall classe 3 = 0.8729**
- Backtesting em evento de maio/2024:
  - **F1-macro = 0.8149**
  - **Recall classe 3 = 0.8919**
  - **Lead time operacional: 240 horas** â€” o modelo emitiu alertas CRÃTICO sequenciais hora a hora desde 01/05, detectando o inÃ­cio da rampa de degradaÃ§Ã£o 10 dias antes do pico de Kp=9 em 11/05 (horizonte de prediÃ§Ã£o: t+1h)

## Arquitetura

```text
Dados NOAA/OMNIWeb
    â†“
Ingestion + Feature Engineering
    â†“
IPO (interno)
    â†“
XGBoost
    â†“
OGII (operacional)
    â†“
Dashboard Streamlit (calcula/visualiza OGII)
    â†“
risk_scores (SQLite) â† Ãºltima inferÃªncia operacional no modo normal
    â†“
ingestion/mqtt_telemetry.py
    â†“
orbitalshield/alerts  â†’  ESP32 (orbital_shield.ino)
                                â†“
orbitalshield/esp32/telemetry  â†’  ingestion/mqtt_telemetry.py
                                â†“
                          esp32_telemetry (SQLite)
```

## Stack

- Python 3.11
- XGBoost
- Streamlit
- SQLite + SQLAlchemy
- MQTT (Paho)
- ESP32 + Arduino IDE

## Estrutura do repositÃ³rio

```text
orbitalshield_gs/
â”œâ”€â”€ backtesting/
â”‚   â”œâ”€â”€ backtest_may2024.py
â”‚   â””â”€â”€ results/
â”œâ”€â”€ dashboard/
â”‚   â””â”€â”€ app.py
â”œâ”€â”€ data/
â”‚   â””â”€â”€ reports/
â”œâ”€â”€ db/
â”‚   â”œâ”€â”€ connection.py
â”‚   â””â”€â”€ models.py
â”œâ”€â”€ esp32/
â”‚   â”œâ”€â”€ orbital_shield.ino
â”‚   â””â”€â”€ README.md
â”œâ”€â”€ experiments/
â”œâ”€â”€ features/
â”‚   â”œâ”€â”€ engineering.py
â”‚   â””â”€â”€ ipo.py
â”œâ”€â”€ ingestion/
â”‚   â”œâ”€â”€ omniweb_loader.py
â”‚   â”œâ”€â”€ noaa_collector.py
â”‚   â””â”€â”€ mqtt_telemetry.py
â”œâ”€â”€ model/
â”‚   â”œâ”€â”€ artifacts/
â”‚   â”œâ”€â”€ train.py
â”‚   â””â”€â”€ predict.py
â”œâ”€â”€ research/
â”‚   â”œâ”€â”€ ipo_definition.md
â”‚   â””â”€â”€ kp_arima_forecast.R
â”œâ”€â”€ sprint0/
â”‚   â”œâ”€â”€ 01_ipo_distribution.py
â”‚   â””â”€â”€ thresholds.json
â”œâ”€â”€ validation/
â”œâ”€â”€ .env.example
â”œâ”€â”€ .gitignore
â”œâ”€â”€ .streamlit/
â”‚   â””â”€â”€ config.toml
â”œâ”€â”€ setup.py
â””â”€â”€ README.md
```

## PrÃ©-requisitos

- Python 3.11+
- R 4.6+ para a projeÃ§Ã£o ARIMA de tendÃªncia Kp
- SQLite via biblioteca padrÃ£o do Python

## InstalaÃ§Ã£o

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

## ExecuÃ§Ã£o

### 1. InicializaÃ§Ã£o do banco
```bash
python -c "from db.connection import init_db; init_db()"
```

### 2. IngestÃ£o de dados
```bash
python ingestion/omniweb_loader.py
python -c "from ingestion.omniweb_loader import load_historical; load_historical(2024, 2024)"
```

### 3. Sprint 0 â€” Gate cientÃ­fico
```bash
python sprint0/01_ipo_distribution.py
```

### 4. Treinamento
```bash
python model/train.py
```

> O arquivo `model/artifacts/xgboost_model.joblib` nÃ£o Ã© versionado no GitHub.
> Ele Ã© gerado localmente por `python model/train.py`.

### 5. Backtesting
```bash
python backtesting/backtest_may2024.py
```

### 6. ProjeÃ§Ã£o ARIMA â€” TendÃªncia Kp 24h (R)
Execute da raiz do projeto:
```bash
Rscript research/kp_arima_forecast.R
```

O script instala/carrega os pacotes R necessÃ¡rios: `forecast`, `RSQLite`, `DBI`, `ggplot2` e `dplyr`.

### 7. Dashboard
```bash
streamlit run dashboard/app.py
```

> No modo normal, o dashboard atualiza a Ãºltima inferÃªncia em `risk_scores`.
> Caso a tabela esteja vazia, o bridge MQTT usa fallback `MODERADO`.
> Para a demo ESP32, abra o dashboard em modo normal antes da bridge MQTT para popular `risk_scores` com um OGII real.

### 8. Bridge MQTT (ESP32 â†” banco â†” ESP32)
```bash
python ingestion/mqtt_telemetry.py
```

### 9. Firmware ESP32
Abra `esp32/orbital_shield.ino` na Arduino IDE.  
Configure `WIFI_SSID` e `WIFI_PASSWORD` no sketch.  
Para demonstraÃ§Ã£o sem hardware fÃ­sico: [Wokwi](https://wokwi.com/projects/new/esp32)

### ObservaÃ§Ã£o sobre artefatos locais

O banco `orbitalshield.db` e o modelo `xgboost_model.joblib` nÃ£o sÃ£o versionados.
Eles sÃ£o recriados pelos passos de inicializaÃ§Ã£o, ingestÃ£o e treinamento para manter
rastreabilidade e evitar versionar arquivos pesados ou sensÃ­veis.

## OrganizaÃ§Ã£o por camadas

### Camada 1 â€” IPO
- DefiniÃ§Ã£o interna do Ã­ndice
- Feature engineering orientado por fÃ­sica de clima espacial
- Thresholds congelados apÃ³s Sprint 0

### Camada 2 â€” Modelo
- Treinamento com XGBoost
- PersistÃªncia de artefatos em `model/artifacts/`
- Metadados de modelo e thresholds versionados

### Camada 3 â€” OGII
- ConversÃ£o da saÃ­da do modelo para Ã­ndice operacional 0â€“100
- ExposiÃ§Ã£o para dashboard, telemetria e integraÃ§Ãµes

### Camada 4 â€” Telemetria ESP32
- NÃ³ IoT que assina `orbitalshield/alerts` via MQTT
- Simula degradaÃ§Ã£o GNSS (HDOP, satÃ©lites, fix) proporcional ao OGII
- Publica `orbitalshield/esp32/telemetry` a cada 5s
- Bridge Python lÃª o Ãºltimo OGII salvo em `risk_scores`, publica alertas MQTT e persiste telemetria em `esp32_telemetry` (SQLite)

## TÃ³picos MQTT

| TÃ³pico | DireÃ§Ã£o | Payload |
|---|---|---|
| `orbitalshield/alerts` | `mqtt_telemetry.py` â†’ ESP32 | `{ "ogii": 82, "level": "CRÃTICO" }` |
| `orbitalshield/esp32/telemetry` | ESP32 â†’ `mqtt_telemetry.py` | `{ "hdop": 5.2, "satellites_visible": 5, ... }` |

## ValidaÃ§Ã£o em trÃªs camadas

| Camada | O que valida | Resultado |
|---|---|---|
| EstatÃ­stica | F1-macro, recall crÃ­tico no test set | 0.8149 / 0.8919 |
| Operacional | OGII + recomendaÃ§Ã£o RTK no dashboard | AntecipaÃ§Ã£o de 240h |
| Proxy fÃ­sico | HDOP e satÃ©lites via ESP32 | CorrelaÃ§Ã£o com alert_level |

## Integrantes do grupo

| Nome | E-mail | RM |
|---|---|---|
| Lucas Carvalho Cordeiro | carvalho.lucascc@gmail.com | 570388 |
| Larissa da Silva Marcelino | larissamarcelinocpb@gmail.com | 571790 |
| Abner Henrique Dias Rosa Sanches | abner.mtpvp@gmail.com | 572253 |
| Brenoezo Leardini | b.leardini@gmail.com | 572533 |
| Elton Modesto de Souza Dias | elton.redes@hotmail.com | 572530 |


## SeguranÃ§a

O projeto implementa prÃ¡ticas de seguranÃ§a em mÃºltiplas camadas:

### ProteÃ§Ã£o de credenciais
- VariÃ¡veis sensÃ­veis (broker MQTT, caminhos, chaves) isoladas em `.env`
- `.env` protegido pelo `.gitignore` â€” nunca versionado
- `.env.example` documenta as variÃ¡veis sem expor valores reais
- ValidaÃ§Ã£o de variÃ¡veis obrigatÃ³rias no startup via `db/connection.py`

### SeparaÃ§Ã£o de camadas
- IPO Ã© constructo interno â€” nÃ£o exposto na interface ou em logs
- OGII calculado exclusivamente em `model/predict.py`
- Artefatos do modelo (`.joblib`) no `.gitignore` â€” nÃ£o versionados

### IoT / MQTT
- Credenciais do broker via `.env` (nunca hardcoded em produÃ§Ã£o)
- Payload ESP32 com `is_replay: true` â€” transparÃªncia de dados simulados
- TÃ³picos com namespace dedicado (`orbitalshield/`)

### Dados e rastreabilidade
- Test set maio/2024 usado uma Ãºnica vez â€” resultados congelados
- Thresholds versionados em `sprint0/thresholds.json`
- Banco SQLite local â€” dados nÃ£o expostos a serviÃ§os externos

### PrÃ³ximos passos de seguranÃ§a (fase 2)
- TLS no broker MQTT (porta 8883)
- AutenticaÃ§Ã£o username/password no broker
- Rate limiting no dashboard para deploy pÃºblico

## Roadmap â€” ExtensÃµes Futuras

O OrbitalShield foi concebido para agricultura de precisÃ£o, mas o problema do clima espacial Ã© mais amplo. O grupo identificou duas extensÃµes naturais do sistema:

### ExtensÃ£o 1 â€” ValidaÃ§Ã£o RBMC/IBGE (SunStrike)

A Rede Brasileira de Monitoramento ContÃ­nuo GPS do IBGE registrou deriva de posicionamento de atÃ© **8,2 metros** na estaÃ§Ã£o CUIB (CuiabÃ¡/MT) durante a tempestade de maio/2024. Esses dados RINEX sÃ£o pÃºblicos e representam o **ground truth real** de degradaÃ§Ã£o GNSS em solo brasileiro.

A prÃ³xima versÃ£o do OrbitalShield integrarÃ¡ os dados RBMC/IBGE como validaÃ§Ã£o fÃ­sica direta â€” substituindo o proxy ESP32 por mediÃ§Ãµes reais de receptor geodÃ©sico, eliminando a principal limitaÃ§Ã£o cientÃ­fica atual.

### ExtensÃ£o 2 â€” OrbitalShield Rural (ConnectWindow)

Mais de **18 milhÃµes de brasileiros** em regiÃµes remotas (comunidades ribeirinhas, quilombolas, agricultores familiares) dependem de satÃ©lites tanto para GPS quanto para comunicaÃ§Ã£o. Nessas regiÃµes, a janela de sinal satelital Ã© intermitente â€” e ninguÃ©m avisa quando ela chega.

A extensÃ£o ConnectWindow integra ao OrbitalShield:

- **Simulador orbital** â€” calcula quando um satÃ©lite passa sobre uma coordenada usando dados TLE do Celestrak/NASA
- **Fila inteligente** â€” prioriza mensagens por urgÃªncia (emergÃªncia mÃ©dica â†’ alerta climÃ¡tico â†’ dados agrÃ­colas â†’ comunicaÃ§Ã£o pessoal)
- **Otimizador de janela** â€” dado X minutos de sinal e banda limitada, decide o que enviar primeiro
- **Preditor com ML** â€” regressÃ£o linear treinada com histÃ³rico de janelas para prever duraÃ§Ã£o e qualidade do prÃ³ximo sinal

**IntegraÃ§Ã£o com o OGII:** quando o Ã­ndice indica risco CRÃTICO E a janela de comunicaÃ§Ã£o Ã© curta, o sistema prioriza automaticamente mensagens de emergÃªncia mÃ©dica e alertas de desastre â€” tecnologia espacial como ferramenta de inclusÃ£o digital.

### Impacto social ampliado

| PÃºblico | Problema | SoluÃ§Ã£o |
|---|---|---|
| Agricultor de precisÃ£o | GPS degrada sem aviso | OGII + alerta 240h de antecipaÃ§Ã£o |
| Agricultor familiar remoto | NÃ£o sabe quando o sinal chega | ConnectWindow â€” fila priorizada |
| Comunidades ribeirinhas | EmergÃªncias mÃ©dicas sem comunicaÃ§Ã£o | Mensagens de saÃºde priorizadas na janela |
| Gestores de desastre | Alertas de enchente nÃ£o chegam | Alertas climÃ¡ticos no topo da fila |

## ObservaÃ§Ãµes importantes

- NÃ£o versionar artefatos pesados ou arquivos sensÃ­veis.
- NÃ£o expor o IPO na interface de usuÃ¡rio.
- NÃ£o recalibrar thresholds fora do processo formal do Sprint 0.
- NÃ£o tratar AMAS como causalidade comprovada.
- Payloads ESP32 com `is_replay: true` sÃ£o dados de demonstraÃ§Ã£o â€” nÃ£o mediÃ§Ã£o real de campo.

## LicenÃ§a

Uso acadÃªmico interno, conforme regras do projeto e da FIAP.

