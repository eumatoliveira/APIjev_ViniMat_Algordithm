# O Gate Matemático e Otimização de Custos (Prompt Mestre)

## 1. O Gate Matemático $S$

Toda requisição recebe um score numérico $S$ antes de qualquer decisão de invocar uma API de inteligência artificial:

$$S = w_N \cdot N + w_R \cdot R + w_U \cdot U + w_Q \cdot Q - w_H \cdot H$$

Onde as variáveis de entrada são normalizadas no intervalo $[0.0, 1.0]$:
- $N$ (**Novelty / Novidade**): Mensura se a pergunta traz entidades, contextos ou termos inéditos não encontrados no vocabulário ou na base local.
- $R$ (**Risk / Risco**): Avalia criticidade jurídica, financeira, segurança ou conformidade.
- $U$ (**Uncertainty / Incerteza**): Grau de ambiguidade ou carência de parâmetros determinísticos.
- $Q$ (**Complexity / Complexidade**): Medida pela extensão do texto, número de cláusulas gramaticais subordinadas e operadores lógicos.
- $H$ (**Cache Hit Probability**): Probabilidade a priori de reutilizar um resultado conhecido ou responder por FAQ.

### Calibração de Pesos Padrão
$$w_N = 0.25, \quad w_R = 0.25, \quad w_U = 0.20, \quad w_Q = 0.20, \quad w_H = 0.20$$
Com limiar de corte inicial:
$$\tau = 0.50$$

### Regra de Decisão
- Se $S < \tau \implies$ **RESOLVER LOCALMENTE**. Custo financeiro de inferência: $\text{US\$}~0.00$.
- Se $S \ge \tau \implies$ **CONSIDERAR ROTEAMENTO DE IA**.

---

## 2. A Função Objetivo de Custo Total $J$

A engenharia FinOps não minimiza apenas chamadas brutas, mas otimiza a função multiobjetivo:

$$\min J = \alpha C + \beta E + \gamma L + \delta R$$

Sujeito a:
- $\text{Qualidade} \ge Q_{\min}$
- $\text{Segurança} \ge S_{\min}$
- $\text{Latência} \le L_{\max}$
- $\text{Risco} \le R_{\max}$

Onde:
- $C$: Custo financeiro direto (tokens de entrada $\times C_{\text{in}}$ + tokens de saída $\times C_{\text{out}}$).
- $E$: Taxa de erro empírico ou falha de validação de schema.
- $L$: Latência de ponta a ponta em segundos.
- $R$: Nível de exposição ao risco.

---

## 3. Modelo de Consumo de Memória (RAM)

Para evitar Out-Of-Memory (OOM) e garantir operação em hardware modesto (~550 MB):

$$\text{RAM} \approx B + k \cdot R + Q + C$$

Onde:
- $B$: Memória base do processo backend (~200 MB).
- $k$: Número de requisições simultâneas em execução ($k \le \text{MAX\_CONCURRENCY}$).
- $R$: Memória média alocada por requisição (~20 MB).
- $Q$: Memória de buffers e filas de streaming (~50 MB).
- $C$: Cache residente em memória RAM LRU (~100 MB).

O sistema aplica **Backpressure** estrito: requisições além de $\text{MAX\_CONCURRENCY} + \text{MAX\_QUEUE}$ são rejeitadas de forma controlada ou respondidas via fallback de emergência local.
