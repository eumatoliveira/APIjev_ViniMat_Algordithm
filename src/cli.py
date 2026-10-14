"""Interactive CLI for Language-as-Calculation & Selective Economic AI Architecture.
CLI Interativo para Linguagem como Calculo e Arquitetura de IA Seletiva e Economica.
"""
import sys
import argparse
import json
import time

from src.storage.persistent_db import PersistentDatabase
from src.data_hub.catalog import list_all_resources, get_catalog_categories, get_resources_by_category, get_resource_by_id
from src.data_hub.downloader import PublicDataDownloader
from src.core_math.ascii_tensor import text_to_ascii_table, text_to_matrix_2d, matrix_to_ascii_grid_string, calculate_ascii_stats
from src.core_math.entropy_shannon import calculate_character_entropy, calculate_word_entropy, calculate_surprisal_profile, calculate_text_compressibility_estimate
from src.core_math.markov_calculator import MarkovTextCalculator
from src.core_math.question_analyzer import analyze_question
from src.core_math.jev_engine import JevSystemOneClassifier, SubMillisecondOutputChecker, FeedbackLearner
from src.vision.image_to_ascii import generate_synthetic_image_matrix, luminance_matrix_to_ascii, apply_sobel_edge_filter, calculate_spatial_gradients
from src.router.gate import MathematicalGate
from src.router.cache import MultiLevelCache
from src.router.llm_adapter import LLMAdapter
from src.router.model_router import ModelRouter
from src.router.finops_telemetry import FinOpsTelemetry


def print_banner():
    banner = r"""
========================================================================================
   _        _   _  _ _____ _   _   _   ____ _____    _   ____    __  __    _  _____ _   _ 
  | |      / \ | \| | ____| | | | / \ / ___| ____|  / \ / ___|  |  \/  |  / \|_   _| | | |
  | |     / _ \| .` |  _| | | | |/ _ \| |  _|  _|   / _ \\___ \  | |\/| | / _ \ | | | |_| |
  | |___ / ___ \ |\ | |___| |_| / ___ \ |_| | |___ / ___ \___) | | |  | |/ ___ \| | |  _  |
  |_____/_/   \_\_| \_____|\___/_/   \_\____|_____/_/   \_\____/  |_|  |_/_/   \_\_| |_| |_|
        DETERMINISTIC LANGUAGE-AS-CALCULATION & SELECTIVE ECONOMIC AI (PROMPT MESTRE)
========================================================================================
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    print(banner)


def cmd_catalog(args, db: PersistentDatabase):
    """List or filter available public domain books, corpora, and lexicons."""
    print("\n[CATALOGO DE DADOS PUBLICOS E CORPORA ABERTOS]")
    categories = get_catalog_categories()
    
    for cat in categories:
        print(f"\n--- {cat} ---")
        items = get_resources_by_category(cat)
        for it in items:
            print(f" * ID: {it['id']}")
            print(f"   Titulo: {it['title']} ({it['author']}) | Idioma: {it['language'].upper()} | Tam: ~{it['estimated_size_kb']} KB")
            print(f"   Descricao: {it['description']}")


def cmd_download(args, db: PersistentDatabase):
    """Download a selected public corpus or seed offline memory."""
    downloader = PublicDataDownloader(db=db)
    
    if args.resource_id == "seed_all":
        print("\n[INGESTAO DE MEMORIA PERSISTENTE SEMENTE OFFLINE]")
        res = downloader.ingest_offline_seed_corpus()
        print(f"[OK] Sucesso: {res['message']}")
        return

    print(f"\n[BAIXANDO RECURSO PUBLICO: {args.resource_id}]")
    res = downloader.download_resource(args.resource_id)
    if res["status"] == "SUCCESS":
        print(f"[OK] Download e Ingestao concluidos com sucesso!")
        print(f"  Bytes baixados: {res['bytes_downloaded']} bytes (streaming em chunks de 64 KB)")
        print(f"  SHA-256: {res['sha256']}")
        print(f"  Entropia de Shannon (amostra): {res['entropy_bits']} bits")
        print(f"  Indexado no SQLite FTS5 (Tabela Virtual de Memoria Persistente)")
    else:
        print(f"[FALHA] no download: {res['message']}")
        print("  Dica: Use '--resource_id seed_all' para injetar a memoria persistente offline.")


def cmd_matrix(args, db: PersistentDatabase):
    """Transform text into ASCII table and 2D matrix heatmap."""
    text = args.text
    print(f"\n[TRANSFORMAÇÃO DE TEXTO EM TABELA E MATRIZ ASCII]")
    print(f"Texto de entrada ({len(text)} caracteres): \"{text}\"")
    
    table = text_to_ascii_table(text)
    print("\n--- TABELA ASCII POSICIONAL (PRIMEIROS 10 CARACTERES) ---")
    print(f"{'Idx':<4} | {'Char':<4} | {'ASCII':<6} | {'Hex':<6} | {'Binário':<10} | {'Normalizado':<11} | {'Categoria':<12}")
    print("-" * 65)
    for row in table[:10]:
        print(f"{row['index']:<4} | {row['char']:<4} | {row['ascii_code']:<6} | {row['hex']:<6} | {row['binary']:<10} | {row['normalized']:<11} | {row['category']:<12}")
        
    stats = calculate_ascii_stats(text)
    print(f"\nEstatísticas de Bytes: Média={stats['mean']} | Desvio={stats['std_dev']} | Mín={stats['min']} | Máx={stats['max']}")
    
    matrix_2d = text_to_matrix_2d(text, grid_width=16)
    grid_art = matrix_to_ascii_grid_string(matrix_2d)
    print("\n--- GRADE DE DENSIDADE MATRICIAL 2D ---")
    print(grid_art)


def cmd_entropy(args, db: PersistentDatabase):
    """Compute Shannon Entropy and information surprisal profile."""
    text = args.text
    print(f"\n[TEORIA DA INFORMAÇÃO & ENTROPIA DE SHANNON (1948)]")
    
    char_h = calculate_character_entropy(text)
    word_h = calculate_word_entropy(text)
    comp = calculate_text_compressibility_estimate(text)
    
    print(f"Entropia por Caractere : {char_h['entropy_bits']} bits (Máx Teórico: {char_h['max_possible_entropy']} bits)")
    print(f"Eficiência Normalizada : {round(char_h['normalized_entropy'] * 100, 1)}%")
    print(f"Entropia por Palavra   : {word_h['word_entropy_bits']} bits (Total: {word_h['total_words']} palavras)")
    print(f"Taxa de Compressão Est.: {comp['estimated_compression_ratio']}x ({comp['theoretical_min_bits_per_char']} bits/char)")
    
    surprisal = calculate_surprisal_profile(text)
    if surprisal:
        print("\n--- PERFIL DE SURPRESA (AUTO-INFORMAÇÃO) DOS PRIMEIROS TOKENS ---")
        for item in surprisal[:5]:
            print(f" • Token: '{item['token']}' | Prob: {item['probability']} | Surpresa: {item['surprisal_bits']} bits")


def cmd_markov(args, db: PersistentDatabase):
    """Train Markov transition matrix and generate words as calculation."""
    corpus = args.corpus or "linguagem como calculo e matriz deterministica de probabilidade"
    seed = args.seed or "lin"
    
    print(f"\n[CADEIA DE MARKOV & GERAÇÃO ESTOCÁSTICA DETERMINÍSTICA]")
    calc = MarkovTextCalculator(order=int(args.order))
    calc.fit(corpus)
    
    generated = calc.generate(seed=seed, max_length=int(args.length), deterministic=True)
    log_likelihood = calc.calculate_sequence_log_likelihood(generated)
    
    print(f"Corpus de treino : \"{corpus[:60]}...\" ({len(corpus)} caracteres)")
    print(f"Ordem N-Grama    : {args.order}")
    print(f"Semente (Seed)   : \"{seed}\"")
    print(f"Texto Gerado     : \"{generated}\"")
    print(f"Log-Likelihood   : {log_likelihood}")


def cmd_questions(args, db: PersistentDatabase):
    """Deterministically analyze question structure without LLM."""
    q = args.question
    print(f"\n[ANALISADOR DETERMINÍSTICO DE PERGUNTAS (SEM LLM)]")
    analysis = analyze_question(q)
    
    print(f"Pergunta              : \"{analysis['text']}\"")
    print(f"É uma pergunta?       : {'Sim' if analysis['is_question'] else 'Não'}")
    print(f"Operador Interrogativo: {analysis['interrogative'] or 'Nenhum'}")
    print(f"Tipo de Resposta Esp. : {analysis['expected_answer_type'] or 'Declaração'}")
    print(f"Score de Complexidade : {analysis['complexity_score']} (Q)")


def cmd_image(args, db: PersistentDatabase):
    """Generate image matrix, apply Sobel filter, and render ASCII."""
    print(f"\n[PROCESSAMENTO DE IMAGEM PARA MATRIZ ASCII & CONVOLUÇÃO ESPACIAL]")
    mat = generate_synthetic_image_matrix(width=32, height=14, pattern=args.pattern)
    ascii_art = luminance_matrix_to_ascii(mat)
    
    print("\n--- RENDERIZAÇÃO POR DENSIDADE DE LUMINÂNCIA ---")
    print(ascii_art)
    
    edges = apply_sobel_edge_filter(mat)
    edge_art = luminance_matrix_to_ascii([[int(min(255, v)) for v in row] for row in edges], density_ramp=" .*#")
    print("\n--- MAPA DE BORDAS POR CONVOLUÇÃO DE SOBEL (3x3) ---")
    print(edge_art)
    
    stats = calculate_spatial_gradients(mat)
    print(f"\nGradientes Espaciais: Máx={stats['max_gradient']} | Média={stats['mean_gradient']}")


def cmd_route(args, db: PersistentDatabase):
    """Execute the full Prompt Mestre selective economic pipeline on a query."""
    adapter = LLMAdapter()
    gate = MathematicalGate(tau=0.50)
    cache = MultiLevelCache(db=db)
    router = ModelRouter(db=db, llm_adapter=adapter, gate=gate, cache=cache)
    
    query = args.query
    print(f"\n========================================================")
    print(f"[PIPELINE DE IA SELETIVA & ECONÔMICA (PROMPT MESTRE)]")
    print(f"Entrada: \"{query}\"")
    print(f"========================================================")
    
    decision = router.process_request(query)
    
    print(f"\n1. DECISÃO DE ROTA:")
    print(f"   Nível Selecionado     : {decision['level'].value}")
    print(f"   Gate Score (S)        : {decision.get('gate_score', 'N/A')}")
    print(f"   Veio do Cache (L1/L2) : {'SIM (Custo Zero)' if decision['cache_hit'] else 'NÃO'}")
    print(f"   Double Check Ativado  : {'SIM' if decision['double_check_performed'] else 'NÃO'}")
    print(f"   Alerta PGR / RH       : {'ATIVADO (Protocolo Aberto)' if decision.get('pgr_alert') else 'NORMAL'}")
    if decision.get("themes"):
        print(f"   Temas PluraTalks      : {', '.join(decision['themes'])}")
    if decision.get("signals"):
        active_sigs = [k for k, v in decision['signals'].items() if v]
        if active_sigs:
            print(f"   Sinais de Proteção    : {', '.join(active_sigs)}")
    print(f"   Latência de Resposta  : {decision['latency_ms']} ms")
    print(f"   Custo Estimado        : US$ {decision['cost_usd']:.6f} (~ R$ {decision['cost_usd'] * 5.60:.4f})")
    print(f"\n2. RESPOSTA ENTREGUE:")
    print(f"   {decision['text']}")
    print(f"\nID da Requisição para Feedback: {decision['request_id']}")


def cmd_chat(args, db: PersistentDatabase):
    """Start an interactive real-time conversational REPL for MATROUTE-AI."""
    adapter = LLMAdapter()
    gate = MathematicalGate(tau=0.50)
    cache = MultiLevelCache(db=db)
    router = ModelRouter(db=db, llm_adapter=adapter, gate=gate, cache=cache)
    learner = FeedbackLearner()
    
    print("\n" + "="*70)
    print("   MATROUTE-AI — TERMINAL INTERATIVO INTELIGENTE & ECONÔMICO")
    print("   Linguagem como Cálculo | PluraTalks L0 | Gemma Local On-Device")
    print("="*70)
    print(" Comandos Rápidos:")
    print("   'sair' ou 'exit'   -> Encerra o terminal interativo")
    print("   'finops'           -> Exibe o painel de economia e métricas")
    print("   'joia' ou '+1'     -> Avalia positivamente a última resposta")
    print("   'deslike' ou '-1'  -> Avalia negativamente e recalibra limiar tau")
    print("   'gemma: <texto>'   -> Força geração direta via Gemma local")
    print("="*70)
    print("\nOlá! Sou o MATROUTE-AI — Sistema de IA Seletiva, Econômica e Linguagem como Cálculo.")
    print("Estou online e ouvindo! Digite sua dúvida ou mensagem abaixo:\n")
    
    last_request_id = None
    last_route = "LEVEL_0_LOCAL"
    
    while True:
        try:
            user_input = input("MATROUTE-AI > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Encerrando MATROUTE-AI... Até logo!]")
            break
            
        if not user_input:
            continue
            
        cmd_lower = user_input.lower()
        if cmd_lower in ("sair", "exit", "quit", ":q"):
            print("[Encerrando MATROUTE-AI... Sessão finalizada.]")
            break
            
        if cmd_lower == "finops":
            cmd_finops(args, db)
            continue
            
        if cmd_lower in ("joia", "+1", "like", "thumbs_up"):
            if not last_request_id:
                print("[!] Nenhuma requisição recente para avaliar.")
                continue
            learner.record_feedback(last_request_id, True, last_route)
            db.record_feedback(last_request_id, True, last_route, learner.tau)
            print(f"[👍 Joia Registrada] Limiar Tau recalibrado para: {learner.tau:.3f}")
            continue
            
        if cmd_lower in ("deslike", "-1", "dislike", "thumbs_down"):
            if not last_request_id:
                print("[!] Nenhuma requisição recente para avaliar.")
                continue
            learner.record_feedback(last_request_id, False, last_route)
            db.record_feedback(last_request_id, False, last_route, learner.tau)
            print(f"[👎 Deslike Registrado] Limiar Tau ajustado para: {learner.tau:.3f}")
            continue
            
        # Check if direct Gemma invocation was requested
        if user_input.lower().startswith("gemma:"):
            gemma_prompt = user_input[6:].strip()
            print("\n[Roteado para IA Gemma Local (127.0.0.1:13305) - Streaming]")
            def print_token(t: str):
                sys.stdout.write(t)
                sys.stdout.flush()
            res = adapter.generate_local_gemma_stream(prompt=gemma_prompt, on_token=print_token)
            print(f"\n[Gemma Concluído | {res['latency_ms']} ms | Tokens: {res['output_tokens']} | Custo: US$ 0.00]\n")
            last_request_id = f"gemma_{int(time.time())}"
            last_route = "LEVEL_1_SMALL_LOCAL"
            continue
            
        # Standard selective routing
        decision = router.process_request(user_input)
        last_request_id = decision["request_id"]
        last_route = decision["level"].value
        
        route_badge = decision['level'].value
        latency = decision['latency_ms']
        cost = decision['cost_usd']
        themes = f" | Temas: {', '.join(decision['themes'])}" if decision.get("themes") else ""
        
        print(f"\n[{route_badge} | {latency} ms | Custo: US$ {cost:.6f}{themes}]")
        print(decision["text"])
        print()


def cmd_feedback(args, db: PersistentDatabase):
    """Record thumbs-up or thumbs-down feedback and recalibrate routing weights."""
    learner = FeedbackLearner()
    thumbs_up = bool(args.thumbs_up.lower() in ("true", "1", "yes", "joia", "+1"))
    
    learner.record_feedback(
        request_id=args.request_id,
        thumbs_up=thumbs_up,
        route_used=args.route or "L0_LOCAL"
    )
    db.record_feedback(
        request_id=args.request_id,
        thumbs_up=thumbs_up,
        route=args.route or "L0_LOCAL",
        updated_tau=learner.tau
    )
    
    stats = learner.get_learning_stats()
    print(f"\n[FEEDBACK REGISTRADO COM SUCESSO]")
    print(f"Avaliação       : {'👍 Joia (+1)' if thumbs_up else '👎 Deslike (-1)'}")
    print(f"Novo Limiar Tau : {stats['current_tau']} (recalibrado dinamicamente)")
    print(f"Taxa de Sucesso : {stats['positive_ratio'] * 100:.1f}%")


def cmd_finops(args, db: PersistentDatabase):
    """Display Executive FinOps Dashboard comparing Naive vs Optimized."""
    telemetry = FinOpsTelemetry()
    logs = db.get_recent_audit_logs(limit=100)
    dash = telemetry.generate_dashboard_metrics(logs)
    
    print("\n========================================================")
    print("             PAINEL EXECUTIVO FINOPS & ROUTING          ")
    print("========================================================")
    print(f"Total de Requisições Auditadas : {dash['total_requests']}")
    print(f"Resolvidas Localmente (L0/L1)  : {dash['local_resolved_count']} ({dash['local_resolved_pct']}%)")
    print(f"Atendidas por Cache (L1/L2)    : {dash['cache_hits_count']} ({dash['cache_hit_pct']}%)")
    print(f"Encaminhadas para IA (L2–L5)   : {dash['llm_calls_count']} ({dash['llm_call_pct']}%)")
    print(f"Alertas PGR / RH Detectados    : {dash['pgr_alerts_count']}")
    print("--------------------------------------------------------")
    print(f"Custo Cenário Otimizado        : US$ {dash['optimized_cost_usd']:.4f}")
    print(f"Custo Cenário Ingênuo (100% IA): US$ {dash['naive_cost_usd']:.4f}")
    print(f"Economia Estimada Gerada       : US$ {dash['savings_usd']:.4f} ({dash['savings_pct']}%)")
    print(f"Latência Média                 : {dash['mean_latency_ms']} ms")
    print(f"Latência P95                   : {dash['p95_latency_ms']} ms")
    print(f"Índice de Otimização (J)       : {dash['j_optimization_index']}")
    print("========================================================\n")


def cmd_gemma(args, db: PersistentDatabase):
    """Interact directly with local Gemma model via Lemonade Server with real-time streaming."""
    adapter = LLMAdapter()
    print("\n[VERIFICANDO IA GEMMA LOCAL NO LEMONADE SERVER]")
    is_available = adapter.check_local_gemma_available()
    print(f"Status do Servidor Lemonade (127.0.0.1:13305): {'[ONLINE - CONECTADO]' if is_available else '[OFFLINE]'}")
    
    if not is_available:
        print("\n[AVISO] O servidor Lemonade não está respondendo em http://127.0.0.1:13305.")
        print("Certifique-se de que o Lemonade Server ou o ASRock Claw Quickset está em execução com o modelo Gemma.")
        return

    prompt = args.prompt
    print(f"\nPrompt enviado: \"{prompt}\"")
    print("\n" + "="*65)
    print("                 RESPOSTA DA IA GEMMA LOCAL (STREAMING)        ")
    print("="*65)
    
    def print_token(token: str):
        sys.stdout.write(token)
        sys.stdout.flush()

    res = adapter.generate_local_gemma_stream(
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        on_token=print_token
    )
    
    print("\n" + "="*65)
    print(f"Modelo Utilizado  : {res['model_used']}")
    print(f"Tokens de Entrada : {res['input_tokens']} tokens")
    print(f"Tokens Gerados    : {res['output_tokens']} tokens")
    print(f"Latência Total    : {res['latency_ms']} ms")
    print(f"Custo Financeiro  : US$ {res['cost_usd']:.2f} (100% Gratuito / On-Device)")
    print("="*65 + "\n")


def cmd_web(args, db: PersistentDatabase):
    """Launch the FastAPI Web Interface."""
    import uvicorn
    import webbrowser
    print("\n[INICIANDO SERVIDOR WEB MATROUTE-AI]")
    print("Acesse no navegador: http://127.0.0.1:8000")
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception:
        pass
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=8000, reload=False)


def main():
    print_banner()
    db = PersistentDatabase()
    
    parser = argparse.ArgumentParser(description="Language as Calculation & Selective AI CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Catalog
    subparsers.add_parser("catalog", help="List open public domain corpora and materials")

    # Download
    p_down = subparsers.add_parser("download", help="Download and ingest public corpus into SQLite")
    p_down.add_argument("--resource_id", default="seed_all", help="Catalog ID or 'seed_all'")

    # Matrix
    p_mat = subparsers.add_parser("matrix", help="Transform text into ASCII table and 2D matrix")
    p_mat.add_argument("--text", default="LINGUAGEM COMO CALCULO", help="Input text string")

    # Entropy
    p_ent = subparsers.add_parser("entropy", help="Compute Shannon Entropy and surprisal profile")
    p_ent.add_argument("--text", default="Para redefinir sua senha com seguranca acesse configuracoes", help="Input text")

    # Markov
    p_mar = subparsers.add_parser("markov", help="Train Markov transition matrix and generate words")
    p_mar.add_argument("--corpus", default="inteligencia e calculo deterministico de palavras e caracteres", help="Corpus")
    p_mar.add_argument("--seed", default="int", help="Seed string")
    p_mar.add_argument("--order", default=2, type=int, help="N-gram order")
    p_mar.add_argument("--length", default=30, type=int, help="Max length")

    # Questions
    p_q = subparsers.add_parser("questions", help="Analyze questions deterministically")
    p_q.add_argument("--question", default="Como faco para redefinir minha senha?", help="Question text")

    # Image
    p_img = subparsers.add_parser("image", help="Render image into ASCII and apply Sobel edge filter")
    p_img.add_argument("--pattern", default="circle", choices=["gradient", "horizontal_gradient", "circle", "checkerboard"])

    # Route
    p_route = subparsers.add_parser("route", help="Process request through Prompt Mestre selective pipeline")
    p_route.add_argument("--query", "--prompt", dest="query", default="Como faco para redefinir minha senha?", help="User prompt")

    # Chat (Interactive REPL)
    subparsers.add_parser("chat", help="Start interactive conversational terminal")

    # Web (FastAPI UI)
    subparsers.add_parser("web", help="Launch interactive web testing playground")

    # Feedback
    p_feed = subparsers.add_parser("feedback", help="Record thumbs up/down and recalibrate threshold tau")
    p_feed.add_argument("--request_id", default="req_manual", help="Request ID")
    p_feed.add_argument("--thumbs_up", default="true", help="true or false")
    p_feed.add_argument("--route", default="LEVEL_0_LOCAL", help="Route used")

    # FinOps
    subparsers.add_parser("finops", help="Display Executive FinOps Dashboard")

    # Gemma Local
    p_gemma = subparsers.add_parser("gemma", help="Direct test with local Gemma AI on Lemonade Server")
    p_gemma.add_argument("--prompt", "--query", dest="prompt", default="Explique como ler um idioma como cálculo e não como token.", help="Prompt text")
    p_gemma.add_argument("--max_tokens", default=1500, type=int, help="Max output tokens")
    p_gemma.add_argument("--temperature", default=0.2, type=float, help="Sampling temperature")

    if len(sys.argv) == 1:
        # Default to interactive chat if no arguments passed
        cmd_chat(None, db)
        sys.exit(0)

    args = parser.parse_args()

    cmd_map = {
        "catalog": cmd_catalog,
        "download": cmd_download,
        "matrix": cmd_matrix,
        "entropy": cmd_entropy,
        "markov": cmd_markov,
        "questions": cmd_questions,
        "image": cmd_image,
        "route": cmd_route,
        "chat": cmd_chat,
        "web": cmd_web,
        "feedback": cmd_feedback,
        "finops": cmd_finops,
        "gemma": cmd_gemma,
    }

    handler = cmd_map.get(args.command)
    if handler:
        handler(args, db)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
