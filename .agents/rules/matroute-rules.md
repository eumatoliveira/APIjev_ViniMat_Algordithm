# MATROUTE-AI — Regras Operacionais e Invariantes

## 1. Modelos de Raciocínio e Pensamento Extendido (Gemma / Thinking Models)
- Quando interagir com modelos locais ou remotos que emitem `reasoning_content` (como `Gemma-4-E2B-it-GGUF` no Lemonade Server), configure `max_tokens >= 1500` por padrão.
- O planejamento e a cadeia de pensamento interna nunca devem esgotar a cota de tokens antes da emissão da resposta final executiva.
- Trate o fluxo de raciocínio e a resposta final de forma visualmente desacoplada no terminal e no ambiente web.

## 2. Experiência de Linha de Comando e Interatividade (CLI UX)
- No Windows Command Prompt (`cmd.exe`), comandos de entrada conversacional (`ola.bat`, saudações, `chat.bat`) devem manter o usuário dentro do REPL interativo contínuo (`MATROUTE-AI > `).
- Evitar retorno prematuro ao prompt do shell do sistema operacional para que mensagens naturais seguintes (ex: *"Tudo bem?"*) permaneçam no pipeline da IA e não causem erro de comando inválido no Windows.

## 3. Precedência de Segurança e Proteção Vital sobre Cache
- Avalie sinais de risco vital (violência, assédio, crise e autoagressão) no **Passo 0** com o motor determinístico ([plura_engine.py](file:///C:/Users/mathe/.gemini/antigravity/scratch/language-as-calculation/src/core_math/plura_engine.py)), **antes** de qualquer consulta de cache ou chamada de modelo.
- Em caso de ativação de sinal de proteção, priorize a entrega imediata de canais de apoio (CVV 188, SAMU 192, Canais de Ética/Ouvidoria) com latência sub-milissegundo (< 0.2 ms) e custo zero.
- Respostas cacheadas desatualizadas jamais devem interceptar ou omitir protocolos de segurança.
