# Linguagem como Cálculo — Matrizes ASCII, Entropia e Markov

## 1. Por que Ler Idioma como Cálculo e Não como Token?

Modelos de linguagem convencionais (LLMs) dependem de vocabulários proprietários de 50.000 a 100.000 tokens gerados por Byte-Pair Encoding (BPE) ou SentencePiece. Essa representação cria três problemas centrais:
1. **Opacidade Estatística**: Não é possível inspecionar formalmente as propriedades discretas do texto caractere a caractere.
2. **Custo por Chamada**: Cada token emitido ou lido acarreta latência de streaming e custo financeiro contínuo.
3. **Alucinação**: Como a geração autoregressiva busca maximizar probabilidades locais sem verificação de tipos, respostas aparentemente lógicas podem conter erros factuais fatais.

O paradigma **Linguagem como Cálculo** trata o texto como dados determinísticos manipuláveis por álgebra linear, teoria de autômatos e estocástica discreta.

---

## 2. A Matriz Posicional ASCII

Qualquer texto de comprimento $N$ é expresso como um tensor de posições e bytes:

$$T = [c_0, c_1, \dots, c_{N-1}], \quad c_i \in \{0, \dots, 255\}$$

Para análise bidimensional (ex: detecção de padrões periódicos ou blocos gramaticais), o vetor é dobrado em uma matriz $M \in \mathbb{R}^{R \times W}$ de largura $W$:

$$M_{r, c} = \begin{cases} \frac{c_{r \cdot W + c}}{255.0}, & \text{se } r \cdot W + c < N \\ 0.0, & \text{caso contrário (padding)} \end{cases}$$

---

## 3. Matrizes de Transição de Markov

A probabilidade de um caractere ou palavra $X_{t}$ suceder uma sequência prévia $X_{t-1}, \dots, X_{t-k}$ é dada pela matriz estocástica $P$:

$$P_{ij} = P(X_t = j \mid X_{t-1} = i) = \frac{\text{Count}(i \to j)}{\sum_k \text{Count}(i \to k)}$$

Com a propriedade estocástica estrita:
$$\sum_{j} P_{ij} = 1.0, \quad \forall i$$

Isso viabiliza a geração matemática determinística (via operador $\text{argmax}$) e o cálculo exato da **log-verossimilhança** de uma sentença:
$$\log \mathcal{L}(S) = \sum_{t=1}^{|S|} \log P(s_t \mid s_{t-1})$$

---

## 4. Decomposição Determinística de Perguntas

Em vez de usar redes neurais para adivinhar a intenção de uma pergunta, o módulo `question_analyzer` decompõe a sentença em tuplas formais:
$$\text{Pergunta} \implies \langle \text{Operador}, \text{Foco}, \text{Predicado}, \text{Tipo de Resposta}, Q \rangle$$

- Operador Interrogativo: $\text{Quem} \implies \text{PERSON}$, $\text{Onde} \implies \text{LOCATION}$, $\text{Como} \implies \text{PROCEDURE}$.
- Complexidade Sintática:
  $$Q = \min(0.5, \frac{\text{palavras}}{40}) + \min(0.3, 0.1 \times \text{cláusulas}) + \min(0.2, 0.05 \times \text{conjunções})$$
