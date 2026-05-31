# research/kp_arima_forecast.R
# =============================================================================
# OrbitalShield — Previsão de Tendência Kp com ARIMA (R + forecast)
# =============================================================================
# Objetivo:
#   Complementar o XGBoost (risco t+1h) com uma projeção de tendência
#   do índice Kp para as próximas 24h usando modelo ARIMA.
#
# Camadas do sistema:
#   XGBoost  → risco operacional GNSS em t+1h (classificação)
#   ARIMA    → tendência futura do Kp nas próximas 24h (séries temporais)
#
# Fluxo:
#   1. Instala pacotes se necessário
#   2. Lê Kp histórico do banco SQLite (últimas 720h = 30 dias)
#   3. Ajusta modelo ARIMA automaticamente (auto.arima)
#   4. Projeta próximas 24h com intervalos de confiança
#   5. Exporta CSV e PNG em data/reports/
#
# Uso:
#   Rscript research/kp_arima_forecast.R
#
# Saídas:
#   data/reports/kp_forecast.csv        — projeção horária 24h
#   data/reports/kp_arima_forecast.png  — gráfico série + projeção
#   data/reports/kp_arima_summary.txt   — resumo do modelo
# =============================================================================

cat("╔══════════════════════════════════════════════════╗\n")
cat("║  OrbitalShield — ARIMA Forecast (R 4.6)         ║\n")
cat("║  Tendência Kp próximas 24h                       ║\n")
cat("╚══════════════════════════════════════════════════╝\n\n")

# ─── 1. Instalação automática de pacotes ─────────────────────────────────────

pkgs <- c("forecast", "RSQLite", "DBI", "ggplot2", "dplyr")

for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Instalando pacote: %s ...\n", pkg))
    install.packages(pkg,
                     repos  = "https://cran.r-project.org",
                     quiet  = TRUE,
                     dependencies = TRUE)
  }
}

suppressPackageStartupMessages({
  library(forecast)
  library(RSQLite)
  library(DBI)
  library(ggplot2)
  library(dplyr)
})

cat("✅ Pacotes carregados\n\n")

# ─── 2. Caminho do banco ──────────────────────────────────────────────────────

# Usa o diretório de trabalho atual como raiz do projeto
# Execute sempre da raiz: Rscript research/kp_arima_forecast.R
root_dir   <- getwd()
db_path    <- file.path(root_dir, "orbitalshield.db")
output_dir <- file.path(root_dir, "data", "reports")

if (!file.exists(db_path)) {
  stop(paste("Banco não encontrado:", db_path,
             "\nExecute o script da raiz do projeto."))
}

cat(sprintf("Banco: %s\n", db_path))

# ─── 3. Leitura do banco ──────────────────────────────────────────────────────

con <- dbConnect(RSQLite::SQLite(), db_path)

query <- "
  SELECT collected_at, kp
  FROM space_weather_raw
  WHERE source = 'omniweb'
    AND kp IS NOT NULL
  ORDER BY collected_at DESC
  LIMIT 720
"

raw <- dbGetQuery(con, query)
dbDisconnect(con)

if (nrow(raw) == 0) {
  stop("Sem dados no banco. Execute python ingestion/omniweb_loader.py primeiro.")
}

# Ordena cronologicamente e converte
raw <- raw[order(raw$collected_at), ]
kp_series <- as.numeric(raw$kp)
n_obs     <- length(kp_series)

cat(sprintf("Registros carregados: %d horas (%.0f dias)\n",
            n_obs, n_obs / 24))
cat(sprintf("Kp — min: %.1f  max: %.1f  média: %.2f\n\n",
            min(kp_series), max(kp_series), mean(kp_series)))

# ─── 4. Modelo ARIMA ──────────────────────────────────────────────────────────

cat("Ajustando modelo ARIMA (auto.arima)...\n")

# Série temporal horária
ts_kp <- ts(kp_series, frequency = 24)  # 24h = 1 ciclo diurno

# auto.arima seleciona a ordem ARIMA ótima automaticamente
# stepwise=FALSE e approximation=FALSE = busca mais ampla (mais lento, mais preciso)
model <- auto.arima(
  ts_kp,
  stepwise      = TRUE,   # TRUE para velocidade (demo)
  approximation = TRUE,
  seasonal      = TRUE,
  max.p = 5, max.q = 5,
  max.P = 2, max.Q = 2
)

cat(sprintf("Modelo selecionado: %s\n", as.character(model)))

# ─── 5. Previsão 24h ──────────────────────────────────────────────────────────

horizon <- 24
fc <- forecast(model, h = horizon, level = c(80, 95))

# Limita Kp ao intervalo físico [0, 9]
fc_mean <- pmax(0, pmin(9, as.numeric(fc$mean)))
fc_lo80 <- pmax(0, pmin(9, as.numeric(fc$lower[, 1])))
fc_hi80 <- pmax(0, pmin(9, as.numeric(fc$upper[, 1])))
fc_lo95 <- pmax(0, pmin(9, as.numeric(fc$lower[, 2])))
fc_hi95 <- pmax(0, pmin(9, as.numeric(fc$upper[, 2])))

# Última hora histórica como t=0
last_time <- as.POSIXct(raw$collected_at[nrow(raw)],
                         origin = "1970-01-01", tz = "UTC")
forecast_times <- last_time + (1:horizon) * 3600

cat(sprintf("\nProjeção Kp — próximas %dh:\n", horizon))
cat(sprintf("  t+1h:  Kp=%.2f  [80%% IC: %.2f–%.2f]\n",
            fc_mean[1], fc_lo80[1], fc_hi80[1]))
cat(sprintf("  t+6h:  Kp=%.2f  [80%% IC: %.2f–%.2f]\n",
            fc_mean[6], fc_lo80[6], fc_hi80[6]))
cat(sprintf("  t+12h: Kp=%.2f  [80%% IC: %.2f–%.2f]\n",
            fc_mean[12], fc_lo80[12], fc_hi80[12]))
cat(sprintf("  t+24h: Kp=%.2f  [80%% IC: %.2f–%.2f]\n",
            fc_mean[24], fc_lo80[24], fc_hi80[24]))

# ─── 6. Classificação de risco por horizonte ──────────────────────────────────

classify_kp <- function(kp) {
  ifelse(kp < 3, "BAIXO",
  ifelse(kp < 5, "MODERADO",
  ifelse(kp < 7, "ALTO", "CRÍTICO")))
}

risk_24h <- classify_kp(fc_mean)
cat(sprintf("\nDistribuição de risco projetado (24h):\n"))
print(table(risk_24h))

# ─── 7. Exporta CSV ───────────────────────────────────────────────────────────

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

forecast_df <- data.frame(
  horizon_h    = 1:horizon,
  datetime_utc = format(forecast_times, "%Y-%m-%d %H:%M:%S"),
  kp_forecast  = round(fc_mean, 3),
  kp_lo80      = round(fc_lo80, 3),
  kp_hi80      = round(fc_hi80, 3),
  kp_lo95      = round(fc_lo95, 3),
  kp_hi95      = round(fc_hi95, 3),
  risk_level   = risk_24h
)

csv_path <- file.path(output_dir, "kp_forecast.csv")
write.csv(forecast_df, csv_path, row.names = FALSE)
cat(sprintf("\n✅ CSV salvo: %s\n", csv_path))

# ─── 8. Gráfico ───────────────────────────────────────────────────────────────

# Prepara dados históricos (últimas 72h para o gráfico)
hist_n    <- min(72, n_obs)
hist_df   <- data.frame(
  datetime = as.POSIXct(raw$collected_at[(n_obs - hist_n + 1):n_obs],
                         origin = "1970-01-01", tz = "UTC"),
  kp       = kp_series[(n_obs - hist_n + 1):n_obs],
  tipo     = "Histórico"
)

# Dados de projeção
proj_df <- data.frame(
  datetime = forecast_times,
  kp       = fc_mean,
  tipo     = "Projeção ARIMA"
)

# Ribbon de incerteza
ribbon_df <- data.frame(
  datetime = forecast_times,
  lo95     = fc_lo95,
  hi95     = fc_hi95,
  lo80     = fc_lo80,
  hi80     = fc_hi80
)

# Faixas de risco Kp
risk_bands <- data.frame(
  ymin  = c(0, 3, 5, 7),
  ymax  = c(3, 5, 7, 9),
  label = c("BAIXO", "MODERADO", "ALTO", "CRÍTICO"),
  fill  = c("#27ae6022", "#f39c1222", "#e67e2222", "#c0392b22")
)

# Título com modelo
model_str <- paste0("ARIMA", arimaorder(model)[1], arimaorder(model)[2],
                    arimaorder(model)[3])

p <- ggplot() +
  # Faixas de risco
  geom_rect(data = risk_bands,
            aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
            fill = risk_bands$fill, alpha = 0.4) +
  # Ribbon IC 95%
  geom_ribbon(data = ribbon_df,
              aes(x = datetime, ymin = lo95, ymax = hi95),
              fill = "#58a6ff", alpha = 0.12) +
  # Ribbon IC 80%
  geom_ribbon(data = ribbon_df,
              aes(x = datetime, ymin = lo80, ymax = hi80),
              fill = "#58a6ff", alpha = 0.20) +
  # Série histórica
  geom_line(data = hist_df,
            aes(x = datetime, y = kp, color = "Histórico (72h)"),
            linewidth = 0.9, alpha = 0.9) +
  # Projeção
  geom_line(data = proj_df,
            aes(x = datetime, y = kp, color = "Projeção ARIMA (24h)"),
            linewidth = 1.1, linetype = "dashed") +
  # Linha de separação histórico/futuro
  geom_vline(xintercept = as.numeric(last_time),
             color = "#8b949e", linetype = "dotted", linewidth = 0.8) +
  annotate("text", x = last_time, y = 8.5,
           label = "agora", hjust = -0.2, size = 3, color = "#8b949e") +
  # Limiares
  geom_hline(yintercept = 5, color = "#f39c12",
             linetype = "dashed", linewidth = 0.5, alpha = 0.6) +
  geom_hline(yintercept = 7, color = "#c0392b",
             linetype = "dashed", linewidth = 0.5, alpha = 0.6) +
  # Escalas
  scale_color_manual(
    values = c("Histórico (72h)"      = "#e6edf3",
               "Projeção ARIMA (24h)" = "#58a6ff")
  ) +
  scale_y_continuous(limits = c(0, 9.2), breaks = 0:9) +
  scale_x_datetime(date_labels = "%d/%m %Hh", date_breaks = "12 hours",
                   limits = as.POSIXct(c(min(hist_df$datetime),
                                         max(proj_df$datetime)))) +
  # Labels
  labs(
    title    = sprintf("OrbitalShield — Tendência Kp: Histórico 72h + Projeção ARIMA 24h"),
    subtitle = sprintf("Modelo: %s | Faixas: IC 80%% e IC 95%% | Camada complementar ao XGBoost",
                       model_str),
    x        = "Data/Hora (UTC)",
    y        = "Índice Kp",
    color    = NULL,
    caption  = "Nota: ARIMA projeta tendência do Kp — não substitui o OGII do XGBoost"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background  = element_rect(fill = "#0d1117", color = NA),
    panel.background = element_rect(fill = "#0d1117", color = NA),
    panel.grid.major = element_line(color = "#21262d", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    text             = element_text(color = "#e6edf3"),
    axis.text        = element_text(color = "#8b949e", size = 8),
    plot.title       = element_text(color = "#e6edf3", size = 12,
                                    face = "bold", hjust = 0),
    plot.subtitle    = element_text(color = "#8b949e", size = 9, hjust = 0),
    plot.caption     = element_text(color = "#484f58", size = 8,
                                    hjust = 0, face = "italic"),
    legend.background = element_rect(fill = "#161b22", color = NA),
    legend.text       = element_text(color = "#e6edf3", size = 9),
    legend.position   = "top",
    axis.text.x       = element_text(angle = 30, hjust = 1)
  )

png_path <- file.path(output_dir, "kp_arima_forecast.png")
ggsave(png_path, plot = p, width = 12, height = 6, dpi = 150,
       bg = "#0d1117")
cat(sprintf("✅ Gráfico salvo: %s\n", png_path))

# ─── 9. Resumo do modelo ──────────────────────────────────────────────────────

txt_path <- file.path(output_dir, "kp_arima_summary.txt")
sink(txt_path)
cat("OrbitalShield — ARIMA Forecast Summary\n")
cat("========================================\n\n")
cat(sprintf("Data/hora execução: %s UTC\n", format(Sys.time(), tz = "UTC")))
cat(sprintf("Observações usadas: %d horas\n", n_obs))
cat(sprintf("Horizonte:          24 horas\n\n"))
cat("Modelo ARIMA:\n")
print(summary(model))
cat("\nProjeção Kp (24h):\n")
print(forecast_df[, c("horizon_h", "datetime_utc", "kp_forecast",
                       "kp_lo80", "kp_hi80", "risk_level")])
sink()
cat(sprintf("✅ Resumo salvo: %s\n", txt_path))

cat("\n╔══════════════════════════════════════════════════╗\n")
cat("║  ARIMA concluído com sucesso                     ║\n")
cat("╚══════════════════════════════════════════════════╝\n")
cat("Próximo passo: o dashboard lê kp_forecast.csv automaticamente\n")