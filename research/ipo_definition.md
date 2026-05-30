# Definição técnica do IPO

## 1. Definição

O **IPO (Internal Prediction Index)** é um constructo interno do OrbitalShield para condensar sinais de clima espacial em uma escala contínua de risco antes da etapa de modelagem supervisionada.

O IPO **não é exposto ao usuário final**. Ele existe apenas para:

- composição de features,
- treinamento do modelo,
- análise de distribuição,
- congelamento de thresholds no Sprint 0.

## 2. Componentes

O IPO é composto por três componentes normalizados:

### C1 — Intensidade geomagnética instantânea
```text
C1 = Kp / 9.0
```

**Justificativa:**  
Kp varia de 0 a 9, então a divisão por 9 normaliza o sinal para a faixa aproximada `[0, 1]`, preservando monotonicidade e interpretabilidade.

---

### C2 — Persistência do campo magnético sulward
```text
C2 = (bz_south × southward_duration) / 150.0
```

onde:

- `bz_south` é a magnitude do Bz sulward,
- `southward_duration` é a duração acumulada do trecho sulward.

**Justificativa do denominador 150.0:**  
O denominador foi calibrado para reduzir saturação precoce do componente. A escala 150 mantém o crescimento do termo em faixa útil para os eventos históricos observados e evita que casos moderados dominem o score. A referência de calibração segue a literatura clássica de acoplamento solar vento–magnetosfera associada a Gonzalez et al. (1994).

---

### C3 — Resposta geomagnética composta
```text
C3 = 0.60 × (|Dst| / 300) + 0.40 × (AE / 2000)
```

**Justificativa dos pesos 0.60 / 0.40:**

- `Dst` representa a compressão do anel de corrente e tende a carregar a parcela principal da severidade geomagnética.
- `AE` captura a atividade auroral e complementa a resposta dinâmica do sistema.
- O peso maior de `Dst` privilegia a estabilidade do índice e sua aderência ao comportamento global dos eventos.

**Justificativa dos referenciais:**

- `300 nT` foi adotado como normalizador para `Dst` por cobrir a faixa relevante dos eventos severos do histórico sem esmagar variações intermediárias.
- `2000 nT` para `AE` preserva dispersão útil em eventos intensos.

## 3. Fórmula final

O score do IPO é calculado por média simples dos três componentes:

```text
IPO = (C1 + C2 + C3) / 3
```

### Motivo da média simples

A média aritmética foi escolhida por parcimônia:

- reduz complexidade de parametrização,
- mantém transparência científica,
- evita superajuste de pesos sem evidência forte para calibração diferenciada,
- facilita auditoria e reprodutibilidade.

## 4. Thresholds

Os thresholds do IPO são definidos por percentis do conjunto de treino:

```text
p25, p50, p75
```

### Regra de corte

- classe 0: abaixo de `p25`
- classe 1: entre `p25` e `p50`
- classe 2: entre `p50` e `p75`
- classe 3: acima de `p75`

### Thresholds congelados

Após o Sprint 0, os thresholds foram congelados para garantir:

- comparabilidade temporal,
- estabilidade do rótulo,
- rastreabilidade científica,
- ausência de vazamento de informação do teste.

## 5. Valores aprovados no Sprint 0

- `p25 = 0.0305`
- `p50 = 0.0592`
- `p75 = 0.1053`

## 6. Interpretação operacional

O IPO é um índice interno de ordenação do risco. Ele serve para treinar o classificador e sustentar a interpretação das classes, mas não deve ser apresentado como produto final.

O produto final do sistema é o **OGII**, calculado somente no módulo de inferência.

## 7. Observação metodológica

O AMAS deve permanecer como **hipótese experimental**. Ele pode ser explorado em pesquisa, mas não deve ser descrito como relação causal comprovada sem validação específica adicional.
