# Especificação Arquitetural — Language as Calculation & IA Seletiva

## 1. Princípio Fundamental e Identidade
Este ecossistema opera sob a premissa inquebrável de Engenharia de Software:
> **"NÃO usar um LLM quando matemática, regras determinísticas, cache, busca, banco de dados ou software tradicional conseguem resolver o problema com qualidade suficiente."**

Trata a linguagem humana como um **cálculo numérico posicional, matriz de densidade de bytes ASCII, matriz estocástica de Markov e teoria da informação de Claude Shannon**, em contraste direto com a abordagem ingênua que terceiriza qualquer caractere a um modelo de linguagem opaco e dispendioso.

---

## 2. Diagrama de Roteamento Seletivo (Prompt Mestre + JEV)

```
INPUT
  ↓
NORMALIZAÇÃO & SEGURANÇA
  ↓
REGRAS LOCAIS
  ↓
MULTI-LEVEL CACHE (L1 RAM LRU / L2 SQLite)
  ├── [Cache Hit] ──────────────────────────────────────────→ RESPOSTA LOCAL ($0.00)
  ↓ (Miss)
JEV SYSTEM ONE CLASSIFIER (Non-Autoregressive < 5ms)
  ↓
CÁLCULO DO GATE MATEMÁTICO: S = wN·N + wR·R + wU·U + wQ·Q - wH·H
  ↓
S < τ ?
  ├── [SIM] ───────────────────────────────────────────────→ RESOLUÇÃO LOCAL (SQLite FTS5 / Math)
  └── [NÃO]
        ↓
    MODEL ROUTER (CASCADE & ESCALATION)
        ├── LEVEL 2: Modelo Barato (Flash-Lite)
        ├── LEVEL 3: Modelo Intermediário (Flash)
        └── LEVEL 4 / 5: Modelo Avançado (Pro) + Double Check se Risco >= Alto
        ↓
    SUB-MILLISECOND VALIDATION (< 2ms)
        ├── Filtro de Tom / Emoções (Alerta PGR / RH)
        ├── Sanidade de Entropia de Shannon
        └── Formato Executivo Passo a Passo
        ↓
    RESPOSTA
        ↓
    TELEMETRIA & AUDITORIA APPEND-ONLY
        ↓
    FEEDBACK REINFORCEMENT (👍 Joia / 👎 Deslike recalibra τ e pesos)
```

---

## 3. Os Pilares Matemáticos

### 3.1 Texto e Imagem como Matriz ASCII
- Cada caractere $c_i$ possui ordinal $ASCII(c_i) \in [0, 255]$ e valor normalizado $v_i = \frac{ASCII(c_i)}{255.0}$.
- Dobramento posicional em matriz 2D de dimensão $M \in \mathbb{R}^{R \times C}$ com preenchimento zero-padding.
- Convolução espacial de Sobel $3 \times 3$ sobre imagens sintetizadas ou matrizes de luminância:
  $$G_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix}, \quad G_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{bmatrix}$$
  Magnitude do gradiente: $G = \sqrt{G_x^2 + G_y^2}$.

### 3.2 Teoria da Informação de Shannon (1948)
- Entropia de caracteres e palavras:
  $$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$
- Surpresa / Auto-informação:
  $$I(x) = -\log_2 P(x)$$
- Compressibilidade teórica baseada no Teorema de Codificação de Fonte:
  $$\text{Taxa} = \frac{8.0}{\max(0.1, H(X))}$$

### 3.3 Motor JEV (TypeSafe AI / System One)
- Modelos System One avaliam o estado completo do programa em uma única passagem paralela não-autoregressiva, produzindo decisões estritamente tipadas com acurácia limitada e calibrada:
  $$P(x) = \text{clamp}(A_{base} \cdot e^{-\lambda d}, A_{min}, A_{max})$$
  onde $A_{min} = 0.80$ e $A_{max} = 0.94$. Nunca promete 100% irrealista.
- Detecção instantânea de riscos de trabalho (assédio, agressão) emitindo alerta imediato para supervisão de Recursos Humanos e abertura de Playbook PGR (Programa de Gerenciamento de Riscos).

### 3.4 Otimização de Custo Total FinOps
Função objetivo minimizada:
$$J = \alpha C + \beta E + \gamma L + \delta R$$
onde $C$ é o custo financeiro, $E$ o erro residual, $L$ a latência e $R$ o risco.

---

## 4. Estrutura Modular
- `src/core_math/`: Matrizes ASCII, Entropia de Shannon, Cadeias de Markov, Parser de Perguntas, Motor JEV.
- `src/vision/`: Imagem para matriz ASCII, convolução Sobel e renderização de densidade.
- `src/storage/`: SQLite FTS5, cache L2 persistente, blobs de matrizes e audit log append-only.
- `src/data_hub/`: Catálogo público curado (Project Gutenberg PT/EN/ES, léxicos) e downloader com streaming em chunks de 64 KB.
- `src/router/`: Gate Matemático $S$, MultiLevelCache L1/L2, Memory Guard (backpressure), LLM Adapter (isolamento), Model Router e Telemetria FinOps.
- `src/cli.py`: Interface unificada de linha de comando.
