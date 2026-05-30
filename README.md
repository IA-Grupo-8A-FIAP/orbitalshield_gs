# OrbitalShield

Sistema de previsão de risco GNSS para agricultura de precisão com base em clima espacial.

## Visão geral

O **OrbitalShield** organiza um pipeline de três camadas para transformar dados de clima espacial em uma predição operacional de impacto sobre GNSS:

1. **IPO (constructo interno)**  
   Índice de Previsão Operacional usado apenas na engenharia de atributos e no treinamento. O IPO **não é exposto ao usuário final**.

2. **Modelo preditivo**  
   Classificador treinado com **XGBoost** para mapear o IPO em classes de risco.

3. **OGII (Operational GNSS Impact Index)**  
   Índice operacional calculado **apenas em `model/predict.py`**, em escala de **0 a 100**, para consumo externo.

## Regras científicas do projeto

- O IPO é um constructo interno e não aparece na interface.
- O OGII é calculado somente no módulo de inferência.
- O conjunto de teste de **maio/2024** foi usado **uma única vez** no backtesting final.
- O AMAS é tratado como hipótese experimental; não deve ser apresentado como causalidade.
- Os thresholds foram congelados após o **Sprint 0** e não devem ser recalibrados retroativamente.

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
  - **Antecipação do pico: 240 horas**

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
Dashboard / Integração MQTT / ESP32
```

## Stack

- Python 3.11
- XGBoost
- Streamlit
- SQLite + SQLAlchemy
- MQTT
- ESP32

## Estrutura do repositório

```text
orbitalshield_gs/
├── backtesting/
├── dashboard/
├── data/reports/
├── db/
├── esp32/
├── experiments/
├── features/
├── ingestion/
├── model/
│   └── artifacts/
├── research/
├── sprint0/
├── validation/
├── .env.example
├── .gitignore
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

### 2. Treinamento
```bash
python model/train.py
```

### 3. Backtesting
```bash
python backtesting/backtest_may2024.py
```

### 4. Dashboard
```bash
streamlit run dashboard/app.py
```

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

## Integrantes do grupo

| Nome | E-mail | RM |
|---|---|---|
| Lucas Carvalho Cordeiro | carvalho.lucascc@gmail.com | 570388 |
| Larissa da Silva Marcelino | larissamarcelinocpb@gmail.com | 571790 |
| Abner Henrique Dias Rosa Sanches | abner.mtpvp@gmail.com | 572253 |
| Brenoezo Leardini | b.leardini@gmail.com | 572533 |
| Elton Modesto de Souza Dias | elton.redes@hotmail.com | 572530 |

## Observações importantes

- Não versionar artefatos pesados ou arquivos sensíveis.
- Não expor o IPO na interface de usuário.
- Não recalibrar thresholds fora do processo formal do Sprint 0.
- Não tratar AMAS como causalidade comprovada.

## Licença

Uso acadêmico interno, conforme regras do projeto e da FIAP.
