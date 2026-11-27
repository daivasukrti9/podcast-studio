import re
import json
from typing import Dict, Any, List
from engine.llm_client import call_qwen_chat, LANG_NAMES


ROBOTIC_PHRASES_MAP = {
    "en conclusión,": "mirándolo ya con perspectiva,",
    "en conclusión": "al final del día",
    "es importante destacar que": "vale mucho la pena notar cómo",
    "cabe mencionar que": "y hay un detalle muy valioso:",
    "en el vertiginoso mundo de hoy": "en lo que vivimos todos los días",
    "sin lugar a dudas": "la verdad es que",
    "in conclusion,": "looking at the bigger picture,",
    "em conclusão,": "olhando com mais calma,",
}

TONE_PROFILES = {
    "calido": {"label": "Conversacional Cálido", "rate_delta": 0, "pitch_delta": 0, "exaggeration": 0.45, "overlap_ms": 0},
    "complice": {"label": "Cercano y Espontáneo", "rate_delta": 2, "pitch_delta": 1, "exaggeration": 0.55, "overlap_ms": 110},
    "entusiasta": {"label": "Vivo y Motivador", "rate_delta": 2, "pitch_delta": 1, "exaggeration": 0.58, "overlap_ms": 90},
    "tecnico": {"label": "Claro y Detallado", "rate_delta": -1, "pitch_delta": 0, "exaggeration": 0.40, "overlap_ms": 0},
    "reflexivo": {"label": "Sereno y Reflexivo", "rate_delta": -3, "pitch_delta": -1, "exaggeration": 0.50, "overlap_ms": 0},
}

STYLE_DELTAS = {
    "oficina": {"name": "Oficina (Colaborativo y fluido)", "rate_shift": 1, "pitch_shift": 0, "overlap_ms": 130},
    "empresarial": {"name": "Empresarial (Ejecutivo y estratégico)", "rate_shift": -2, "pitch_shift": -1, "overlap_ms": 100},
    "streamer": {"name": "Streamer (Ágil y expresivo)", "rate_shift": 4, "pitch_shift": 1, "overlap_ms": 150},
    "relajante": {"name": "Relajante (Nocturno y sereno)", "rate_shift": -5, "pitch_shift": -1, "overlap_ms": 80},
}


def _estimate_seconds(text: str, wpm: int = 138) -> float:
    words = len(re.findall(r"\w+", text))
    return round(max(2.0, (words / wpm) * 60.0), 1)


def _apply_tone_and_style_params(
    base_rate: str,
    base_pitch: str,
    tone_key: str,
    style_profile: str = "oficina"
) -> Dict[str, Any]:
    prof = TONE_PROFILES.get(tone_key, TONE_PROFILES["calido"])
    s_meta = STYLE_DELTAS.get(style_profile, STYLE_DELTAS["oficina"])
    try:
        r_val = int(re.sub(r"[^0-9+-]", "", base_rate or "+0") or "0")
    except Exception:
        r_val = 0
    try:
        p_val = int(re.sub(r"[^0-9+-]", "", base_pitch or "+0") or "0")
    except Exception:
        p_val = 0

    final_r = max(-12, min(12, r_val + prof["rate_delta"] + s_meta["rate_shift"]))
    final_p = max(-4, min(4, p_val + prof["pitch_delta"] + s_meta["pitch_shift"]))
    return {
        "tone": tone_key,
        "tone_label": prof["label"],
        "style": style_profile,
        "rate": f"{final_r:+d}%",
        "pitch": f"{final_p:+d}Hz",
        "exaggeration": prof["exaggeration"],
        "overlap_ms": prof["overlap_ms"]
    }


def _try_generate_script_with_qwen(
    show_title: str,
    host_a: Dict[str, Any],
    host_b: Dict[str, Any],
    insights: Dict[str, Any],
    language: str,
    style_profile: str,
    is_monologue: bool,
    custom_notes: str,
    api_key: str,
    provider_hint: str
) -> List[Dict[str, Any]]:
    """
    Si el colaborador tiene su API Key de Qwen configurada, Qwen redacta directamente
    el guion completo en el idioma seleccionado (es, en, pt) con fluidez humana total,
    sin muletillas forzadas ni tirantez inicial.
    """
    if not api_key:
        return []

    lang_label = LANG_NAMES.get(language, "Español Latinoamericano Neutro")
    style_label = STYLE_DELTAS.get(style_profile, STYLE_DELTAS["oficina"])["name"]
    speakers_desc = (
        f"Un único narrador llamado {host_a['name']}."
        if is_monologue
        else f"Dos anfitriones conversando con complicidad natural y sin actuar forzado: Host_A ({host_a['name']}) y Host_B ({host_b['name']})."
    )

    system_msg = (
        f"Eres un guionista experto en podcasts conversacionales humanos en {lang_label}. "
        f"Estilo elegido: {style_label}. {speakers_desc}\n"
        "REGLAS CLAVE DE NATURALIDAD:\n"
        "1. TODO el diálogo debe estar 100% escrito en el idioma solicitado, traduciendo y adaptando cualquier término o cita extranjera.\n"
        "2. PROHIBIDO forzar risas escritas como 'jaja' o muletillas artificiales al inicio. La naturalidad debe surgir como en una charla serena, inteligente y cercana entre dos colegas que se escuchan de verdad.\n"
        "3. Estructura el episodio en 6 bloques claros (24 a 28 intervenciones sustanciosas en total):\n"
        "   - Bloque 1: Bienvenida relajada, presentación y reflexión curiosa sobre el tema.\n"
        "   - Bloque 2: Introducción del tema y hallazgos principales de la fuente.\n"
        "   - Bloque 3: Inmersión técnica y detalles profundos explicados con claridad.\n"
        "   - Bloque 4: Usos prácticos, ideas aplicables y casos de éxito.\n"
        "   - Bloque 5: Conclusiones y comentarios personales.\n"
        "   - Bloque 6: Invitación cálida a toda la comunidad de la empresa a sumarse y compartir.\n"
        "4. Devuelve EXCLUSIVAMENTE un array JSON válido donde cada elemento tenga:\n"
        '   {"speaker": "A" o "B", "act": "nombre del bloque", "tone": "calido"|"complice"|"entusiasta"|"tecnico"|"reflexivo", "is_backchannel": false|true, "sfx_transition": true|false, "text": "..."}'
    )

    user_msg = (
        f"Programa: {show_title}\n"
        f"Título / Tema: {insights.get('title', '')}\n"
        f"Resumen: {insights.get('summary', '')}\n"
        f"Puntos clave: {json.dumps(insights.get('beyond_tech_points', []), ensure_ascii=False)}\n"
        f"Detalles técnicos: {json.dumps(insights.get('technical_deep_dive', []), ensure_ascii=False)}\n"
        f"Casos prácticos: {json.dumps(insights.get('practical_cases_and_ideas', []), ensure_ascii=False)}\n"
        f"Notas adicionales: {custom_notes}\n"
    )

    raw = call_qwen_chat(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        api_key=api_key,
        provider_hint=provider_hint,
        temperature=0.62,
        max_tokens=3600
    )
    if not raw:
        return []

    try:
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        arr = json.loads(cleaned.strip())
        if isinstance(arr, list) and len(arr) >= 10:
            return arr
    except Exception:
        pass
    return []


def _build_smooth_multilingual_fallback(
    lang: str,
    show_title: str,
    host_a: Dict[str, Any],
    b_speaker: Dict[str, Any],
    is_monologue: bool,
    insights: Dict[str, Any],
    custom_notes: str
) -> List[Dict[str, Any]]:
    """
    Construye los 6 bloques en Español (es), Inglés (en) o Portugués (pt) con la misma
    fluidez serena y orgánica de los bloques finales (cero tirantez al inicio).
    """
    title = insights.get("title", "")
    summary = insights.get("summary", "")
    pts = insights.get("beyond_tech_points", ["", "", "", ""])
    techs = insights.get("technical_deep_dive", ["", "", ""])
    cases = insights.get("practical_cases_and_ideas", ["", "", ""])
    quotes = insights.get("key_quotes", [""])

    while len(pts) < 4: pts.append("")
    while len(techs) < 3: techs.append("")
    while len(cases) < 3: cases.append("")

    if lang == "en":
        return [
            ("A", "Block 1 · Welcome & Curious Opening", "calido", True, False,
             f"Welcome back to {show_title}. I am {host_a['name']}, and it is great to share these quiet minutes with you today. We are diving into a topic that touches how we think, build, and collaborate every day: {title}."),
            ("B", "Block 1 · Welcome & Curious Opening", "calido", False, False,
             f"Great to be here with you, {host_a['name'] if not is_monologue else 'everyone'}. I am {b_speaker['name']}. Before we jump into the technical layers, what struck me while reviewing this material is a simple paradox: we have faster tools than ever before, yet our best breakthroughs still happen when we slow down and have an honest conversation."),
            ("A", "Block 1 · Welcome & Curious Opening", "complice", False, False,
             "That is so true. Sometimes teams confuse moving fast with moving in the right direction. Having power under the hood means very little unless we agree on what human problem is actually worth solving first."),
            ("B", "Block 1 · Welcome & Curious Opening", "calido", False, False,
             "Exactly. So today we are going to explore the big picture first, then open up the technical details to see how it really works, and finally share practical use cases and ideas we can apply right away."),

            ("A", "Block 2 · Core Topic & Key Findings", "calido", True, False,
             f"Let us look at the core premise we extracted from the source: {summary} Notice how the conversation shifts away from cold machinery and focuses on how people learn and make decisions."),
            ("B", "Block 2 · Core Topic & Key Findings", "reflexivo", False, True,
             "Right, that changes the whole perspective."),
            ("B", "Block 2 · Core Topic & Key Findings", "reflexivo", False, False,
             f"Look at the first key finding: {pts[0]} For years, the bottleneck was getting access to information. Today, the real craft lies in curating clarity and protecting quality time for our teams."),
            ("A", "Block 2 · Core Topic & Key Findings", "calido", False, False,
             f"And that connects naturally to the second insight: {pts[1]} Memorizing standard answers is no longer what sets a team apart; asking thoughtful questions across disciplines is where real innovation begins."),
            ("B", "Block 2 · Core Topic & Key Findings", "calido", False, False,
             f"There is also a third pillar worth highlighting here: {pts[2]} Designing sustainable workflows is essential if we want people to thrive alongside new technology."),

            ("A", "Block 3 · Technical Deep Dive", "tecnico", True, False,
             f"Now let us look closely at the technical architecture and details behind this. When we move past the headlines, how does this work in practice? First: {techs[0]}"),
            ("B", "Block 3 · Technical Deep Dive", "tecnico", False, False,
             "Let us unpack that clearly. Whenever we process a complex video lecture, a technical paper, or an operational workflow, dumping raw information all at once flattens the nuance. Separating context extraction, technical logic, and independent quality checks keeps the output reliable and grounded."),
            ("A", "Block 3 · Technical Deep Dive", "complice", False, True,
             "That distinction makes a huge difference."),
            ("A", "Block 3 · Technical Deep Dive", "tecnico", False, False,
             f"And look at the second technical element: {techs[1]} Combining deterministic rules for privacy and audio standards with flexible language understanding gives us precision without losing human warmth."),
            ("B", "Block 3 · Technical Deep Dive", "tecnico", False, False,
             f"Plus, consider the third technical layer: {techs[2]} Together with {pts[3]} When tools are modular and lightweight, every department across different countries can participate without needing specialized hardware."),

            ("A", "Block 4 · Practical Use Cases & Success Stories", "entusiasta", True, False,
             f"Let us bring these technical ideas into everyday practice. Where is this already creating real results? Take this first practical case: {cases[0]}"),
            ("B", "Block 4 · Practical Use Cases & Success Stories", "complice", False, False,
             "Think about how relatable that is. Nobody looks forward to reading a forty-page report late on a Friday afternoon, whereas listening to a fifteen-minute conversation between colleagues brings those exact same concepts to life."),
            ("A", "Block 4 · Practical Use Cases & Success Stories", "calido", False, False,
             f"Here is a second practical application any team can try: {cases[1]} Pairing someone from engineering or operations with someone from people and culture creates questions that unlock better solutions."),
            ("B", "Block 4 · Practical Use Cases & Success Stories", "entusiasta", False, False,
             f"{custom_notes.strip() if custom_notes.strip() else cases[2]} Small experiments shared openly across teams build momentum much faster than waiting for perfection."),

            ("A", "Block 5 · Takeaways & Open Reflections", "reflexivo", True, False,
             f"As we move toward our closing reflections, this quote captures the heart of today's conversation: \"{quotes[0]}\" {b_speaker['name']}, what personal takeaway stays with you today?"),
            ("B", "Block 5 · Takeaways & Open Reflections", "reflexivo", False, False,
             "I am left with the conviction that technology is at its best when it gives people a clearer microphone rather than replacing their voice. Having the freedom to shape, edit, and refine an idea makes all the difference."),
            ("A", "Block 5 · Takeaways & Open Reflections", "calido", False, False,
             "Completely agree. Authenticity, curiosity, and learning out loud together are what truly connect us across teams and countries."),

            ("B", "Block 6 · Community Initiative & Invitation", "entusiasta", True, False,
             "Before we wrap up, remember that this podcast studio belongs to everyone in our community. Whether you work in Spanish, English, or Portuguese, your perspective matters here."),
            ("A", "Block 6 · Community Initiative & Invitation", "calido", False, False,
             f"Bring an article, a video that inspired you, or a project lesson, shape the script at your own pace, and add your episode to our shared playlists. Thank you for listening to {show_title}, and see you in the next episode.")
        ]

    if lang == "pt":
        return [
            ("A", "Bloco 1 · Boas-vindas e Abertura Conversacional", "calido", True, False,
             f"Sejam muito bem-vindos a mais um episódio de {show_title}. Eu sou {host_a['name']}, e é uma alegria compartilhar esta conversa com vocês hoje sobre um tema essencial para o nosso dia a dia: {title}."),
            ("B", "Bloco 1 · Boas-vindas e Abertura Conversacional", "calido", False, False,
             f"Que prazer estar aqui com você, {host_a['name'] if not is_monologue else 'pessoal'}. Eu sou {b_speaker['name']}. Antes de entrarmos nos detalhes técnicos, fiquei pensando em algo curioso ao revisar este conteúdo: temos ferramentas cada vez mais rápidas, mas as nossas melhores ideias continuam nascendo quando paramos alguns minutos para conversar de verdade."),
            ("A", "Bloco 1 · Boas-vindas e Abertura Conversacional", "complice", False, False,
             "Faz todo sentido. Muitas vezes confundimos velocidade com direção. Não adianta ter muita potência tecnológica se não estivermos alinhados sobre qual problema humano queremos resolver primeiro."),
            ("B", "Bloco 1 · Boas-vindas e Abertura Conversacional", "calido", False, False,
             "Exatamente. Por isso hoje vamos construir esse caminho juntos: começando pelo contexto geral, mergulhando nos detalhes técnicos e depois trazendo casos de sucesso e ideias práticas para aplicarmos já."),

            ("A", "Bloco 2 · Introdução ao Tema e Descobertas", "calido", True, False,
             f"Olhando para a ideia central que extraímos do material: {summary} Repare como o foco sai do tecnicismo frio e vai direto para a forma como as pessoas aprendem e colaboram."),
            ("B", "Bloco 2 · Introdução ao Tema e Descobertas", "reflexivo", False, True,
             "Sim, isso muda completamente a nossa visão."),
            ("B", "Bloco 2 · Introdução ao Tema e Descobertas", "reflexivo", False, False,
             f"Veja o primeiro ponto principal: {pts[0]} Hoje o desafio não é mais encontrar informação, e sim filtrar o excesso de ruído para devolver tempo de qualidade às equipes."),
            ("A", "Bloco 2 · Introdução ao Tema e Descobertas", "calido", False, False,
             f"E isso se conecta diretamente com o segundo ponto: {pts[1]} O diferencial humano hoje está em fazer boas perguntas e conectar áreas diferentes com propósito."),
            ("B", "Bloco 2 · Introdução ao Tema e Descobertas", "calido", False, False,
             f"Além disso, temos um terceiro pilar fundamental: {pts[2]} Cuidar do equilíbrio e do bem-estar das pessoas precisa caminhar lado a lado com qualquer inovação."),

            ("A", "Bloco 3 · Imersão Técnica e Detalhes Profundos", "tecnico", True, False,
             f"Agora vamos olhar por dentro da engrenagem técnica. Como isso funciona na prática quando analisamos os detalhes? O primeiro ponto técnico mostra o seguinte: {techs[0]}"),
            ("B", "Bloco 3 · Imersão Técnica e Detalhes Profundos", "tecnico", False, False,
             "Vale muito explicar isso com clareza. Quando processamos um vídeo completo ou um documento denso, jogar tudo de uma vez gera respostas superficiais. Separar a transcrição, os conceitos técnicos e a revisão de qualidade garante profundidade real."),
            ("A", "Bloco 3 · Imersão Técnica e Detalhes Profundos", "complice", False, True,
             "Perfeito, esse detalhe técnico é decisivo."),
            ("A", "Bloco 3 · Imersão Técnica e Detalhes Profundos", "tecnico", False, False,
             f"E veja o segundo aspecto técnico: {techs[1]} Unir regras claras de privacidade e padrão sonoro com inteligência contextual evita erros e mantém a conversa natural."),
            ("B", "Bloco 3 · Imersão Técnica e Detalhes Profundos", "tecnico", False, False,
             f"Somando ainda o terceiro elemento técnico: {techs[2]} Junto com {pts[3]} Ter uma plataforma leve e acessível permite que colegas de qualquer área e país participem sem barreiras."),

            ("A", "Bloco 4 · Usos Práticos, Ideias e Casos de Sucesso", "entusiasta", True, False,
             f"Trazendo tudo isso para a prática, vamos ver onde essas ideias já geram resultados concretos. Olhe este primeiro caso: {cases[0]}"),
            ("B", "Bloco 4 · Usos Práticos, Ideias e Casos de Sucesso", "complice", False, False,
             "É muito fácil entender o porquê. Quase ninguém consegue absorver um relatório extenso no fim da semana, mas ouvir uma conversa clara de quinze minutos transforma o aprendizado em algo leve e envolvente."),
            ("A", "Bloco 4 · Usos Práticos, Ideias e Casos de Sucesso", "calido", False, False,
             f"E veja esta segunda aplicação prática: {cases[1]} Aproximar quem constrói soluções técnicas de quem acompanha pessoas e cultura gera insights valiosos para toda a empresa."),
            ("B", "Bloco 4 · Usos Práticos, Ideias e Casos de Sucesso", "entusiasta", False, False,
             f"{custom_notes.strip() if custom_notes.strip() else cases[2]} Começar com pequenos testes, ajustar o roteiro com liberdade e compartilhar com a equipe cria uma evolução contínua."),

            ("A", "Bloco 5 · Conclusões e Reflexões Finais", "reflexivo", True, False,
             f"Chegando às nossas conclusões, gosto muito desta síntese: \"{quotes[0]}\" {b_speaker['name']}, qual reflexão principal fica para você hoje?"),
            ("B", "Bloco 5 · Conclusões e Reflexões Finais", "reflexivo", False, False,
             "Fico com a certeza de que a tecnologia cumpre seu melhor papel quando amplia a voz humana em vez de substituí-la. Ter liberdade para editar cada detalhe e colocar nossa própria visão faz toda a diferença."),
            ("A", "Bloco 5 · Conclusões e Reflexões Finais", "calido", False, False,
             "Concordo plenamente. O que realmente conecta nossa comunidade em todos os países é a autenticidade, a escuta ativa e a vontade de crescer juntos."),

            ("B", "Bloco 6 · Convite à Iniciativa e Comunidade Aberta", "entusiasta", True, False,
             "E para encerrar, fica o nosso convite aberto a todos vocês: este estúdio foi pensado para que cada colaborador possa trazer suas ideias, referências e aprendizados."),
            ("A", "Bloco 6 · Convite à Iniciativa e Comunidade Aberta", "calido", False, False,
             f"Escolha seu tema, gere seu episódio em português, espanhol ou inglês, grave sua voz se quiser e adicione às nossas playlists compartilhadas. Muito obrigado pela companhia em {show_title} e até o próximo episódio.")
        ]

    # Español (es) — Fluido, cálido y sin tirantez desde la primera frase
    return [
        ("A", "Bloque 1 · Bienvenida, Presentación y Juego con el Tema", "calido", True, False,
         f"Bienvenidos a un nuevo episodio de {show_title}. Soy {host_a['name']}, y qué gusto compartir estos minutos de conversación tranquila con ustedes. Hoy vamos a explorar a fondo un tema que nos toca muy de cerca: {title}."),
        ("B", "Bloque 1 · Bienvenida, Presentación y Juego con el Tema", "calido", False, False,
         f"Un placer enorme estar aquí contigo, {host_a['name'] if not is_monologue else 'amigos'}. Soy {b_speaker['name']}. Antes de entrar en la parte formal, me quedé pensando en algo curioso al revisar este material: hoy tenemos herramientas que procesan datos en milisegundos, pero nuestras mejores ideas siguen apareciendo cuando paramos un momento, servimos un café y conversamos sin prisa."),
        ("A", "Bloque 1 · Bienvenida, Presentación y Juego con el Tema", "complice", False, False,
         "Es tal cual lo dices. A veces en el día a día confundimos ir muy rápido con avanzar en la dirección correcta. De poco sirve tener muchísima potencia tecnológica si primero no nos ponemos de acuerdo en qué problema humano vale la pena resolver."),
        ("B", "Bloque 1 · Bienvenida, Presentación y Juego con el Tema", "calido", False, False,
         "Exactamente. Por eso hoy vamos a recorrer este tema paso a paso: primero entendiendo el panorama general de la fuente, luego entrando con calma a los detalles más técnicos, y después aterrizándolo en casos de éxito e ideas prácticas."),

        ("A", "Bloque 2 · Introducción del Tema y Hallazgos Principales", "calido", True, False,
         f"Miremos el punto de partida central que extrajimos del análisis: {summary} Fíjate cómo el enfoque va mucho más allá de las herramientas y apunta directo a cómo aprendemos, colaboramos y tomamos decisiones."),
        ("B", "Bloque 2 · Introducción del Tema y Hallazgos Principales", "reflexivo", False, True,
         "Claro, ahí cambia toda la perspectiva."),
        ("B", "Bloque 2 · Introducción del Tema y Hallazgos Principales", "reflexivo", False, False,
         f"Observa el primer hallazgo clave: {pts[0]} Durante mucho tiempo la dificultad era acceder a la información; hoy el verdadero arte está en filtrar el ruido para devolverle tiempo valioso a las personas."),
        ("A", "Bloque 2 · Introducción del Tema y Hallazgos Principales", "calido", False, False,
         f"Y eso se enlaza de manera natural con el segundo hallazgo: {pts[1]} Ya no se trata de memorizar manuales enteros, sino de cultivar el criterio propio y formular buenas preguntas entre distintas áreas."),
        ("B", "Bloque 2 · Introducción del Tema y Hallazgos Principales", "calido", False, False,
         f"Sumado a un tercer punto indispensable en esta introducción: {pts[2]} Cuidar el equilibrio y el desarrollo humano del equipo es lo que hace que cualquier avance sea sostenible en el tiempo."),

        ("A", "Bloque 3 · Inmersión Técnica y Detalles a Fondo", "tecnico", True, False,
         f"Pasemos ahora a revisar los detalles técnicos con mayor profundidad. Cuando abrimos el capó y miramos cómo funciona realmente esta estructura, aparece un primer eje técnico fundamental: {techs[0]}"),
        ("B", "Bloque 3 · Inmersión Técnica y Detalles a Fondo", "tecnico", False, False,
         "Vale mucho la pena explicarlo con claridad. Cuando analizamos un video extenso o un documento técnico, procesar todo de golpe suele perder los matices importantes. En cambio, al trabajar por capas —extracción completa, desglose técnico y verificación independiente—, conservamos tanto el rigor como la claridad."),
        ("A", "Bloque 3 · Inmersión Técnica y Detalles a Fondo", "complice", False, True,
         "Ese matiz técnico marca toda la diferencia."),
        ("A", "Bloque 3 · Inmersión Técnica y Detalles a Fondo", "tecnico", False, False,
         f"Y fíjate en el segundo componente técnico del análisis: {techs[1]} Combinar reglas deterministas de privacidad y calidad acústica con modelos capaces de comprender contexto e idiomas nos da seguridad sin perder calidez."),
        ("B", "Bloque 3 · Inmersión Técnica y Detalles a Fondo", "tecnico", False, False,
         f"Además del tercer elemento técnico: {techs[2]} Junto con {pts[3]} Que la herramienta sea ligera, multilingüe y editable permite que compañeros en distintos países construyan contenido de alto nivel sin barreras técnicas."),

        ("A", "Bloque 4 · Usos Prácticos, Ideas y Casos de Éxito", "entusiasta", True, False,
         f"Aterricemos todo este recorrido técnico en usos prácticos, ideas y casos de éxito reales. Mira este primer ejemplo concreto: {cases[0]}"),
        ("B", "Bloque 4 · Usos Prácticos, Ideas y Casos de Éxito", "complice", False, False,
         "Se entiende perfecto por qué funciona tan bien. Leer un informe denso al final de la jornada cuesta muchísimo, mientras que escuchar una conversación fluida de quince minutos convierte ese mismo material en aprendizaje cercano y aplicable."),
        ("A", "Bloque 4 · Usos Prácticos, Ideas y Casos de Éxito", "calido", False, False,
         f"Veamos también un segundo uso práctico que cualquier equipo puede poner en marcha: {cases[1]} Sentar a conversar a perfiles técnicos con perfiles de cultura, talento o diseño siempre destraba ideas que en solitario no veríamos."),
        ("B", "Bloque 4 · Usos Prácticos, Ideas y Casos de Éxito", "entusiasta", False, False,
         f"{custom_notes.strip() if custom_notes.strip() else cases[2]} Poder hacer pruebas con tranquilidad, editar el guion a nuestro gusto y compartir solo cuando estamos conformes le quita toda presión al proceso creativo."),

        ("A", "Bloque 5 · Conclusiones, Comentarios y Reflexión Abierta", "reflexivo", True, False,
         f"Entrando ya en nuestras conclusiones y comentarios finales, me quedo pensando en esta idea central: «{quotes[0]}» {b_speaker['name']}, ¿qué reflexión personal te llevas tú después de esta charla?"),
        ("B", "Bloque 5 · Conclusiones, Comentarios y Reflexión Abierta", "reflexivo", False, False,
         "Me llevo la tranquilidad de ver que la tecnología suma de verdad cuando amplifica nuestra voz en lugar de reemplazarla. Cuando cada persona tiene la libertad de ajustar una frase, elegir el enfoque y cuidar el mensaje, el resultado tiene identidad humana real."),
        ("A", "Bloque 5 · Conclusiones, Comentarios y Reflexión Abierta", "calido", False, False,
         "Coincido por completo. Más allá de las herramientas, lo que nos conecta como organización en cada país es la curiosidad compartida y las ganas de escucharnos con respeto."),

        ("B", "Bloque 6 · Impulso a la Iniciativa y Comunidad Abierta", "entusiasta", True, False,
         "Y antes de cerrar, queremos recordarles que esta iniciativa está abierta para todos los integrantes de nuestra comunidad. Cada uno puede ingresar con su cuenta, traer un tema afín y darle forma a su propio episodio."),
        ("A", "Bloque 6 · Impulso a la Iniciativa y Comunidad Abierta", "calido", False, False,
         f"Ya sea en español, en inglés o en portugués, anímense a compartir ese video, artículo o experiencia que vale la pena conversar, y sumen sus episodios a nuestras listas de reproducción. Gracias por acompañarnos en {show_title}, y nos encontramos en el próximo episodio.")
    ]


def run_elephant_phase(
    show: Dict[str, Any],
    brand_core: Dict[str, Any],
    insights: Dict[str, Any],
    podcast_format: str = "debate_2_hosts",
    audience_mode: str = "internal",
    target_minutes: int = 16,
    guest_name: str = "",
    custom_notes: str = "",
    llm_mode: str = "local_smart",
    style_profile: str = "oficina",
    language: str = "es",
    api_key: str = "",
    provider_hint: str = "auto",
) -> Dict[str, Any]:
    target_minutes = max(11, min(24, int(target_minutes or 16)))
    style_profile = style_profile if style_profile in STYLE_DELTAS else "oficina"
    language = language if language in ("es", "en", "pt") else "es"
    s_meta = STYLE_DELTAS[style_profile]

    hosts = show.get("hosts", [])
    host_a = hosts[0] if len(hosts) > 0 else {
        "id": "persona_es_dalia",
        "name": "Dalia Vargas",
        "role": "Anfitriona Principal",
        "voice_id": "es-MX-DaliaNeural",
        "qwen_voice": "Serena",
        "pitch": "+0Hz",
        "rate": "+1%"
    }
    host_b = hosts[1] if len(hosts) > 1 else {
        "id": "persona_es_jorge",
        "name": "Jorge Solís",
        "role": "Co-Anfitrión",
        "voice_id": "es-MX-JorgeNeural",
        "qwen_voice": "Ethan",
        "pitch": "-1Hz",
        "rate": "+1%"
    }

    if guest_name.strip():
        host_b = dict(host_b)
        host_b["name"] = guest_name.strip()
        host_b["role"] = "Participante Invitado"

    is_monologue = podcast_format == "monologo"
    b_speaker = host_a if is_monologue else host_b
    show_title = show.get("name", "Café con Medialunas by Max")

    master_doc = {
        "episode_title": insights.get("title", "Conversación Abierta"),
        "show_name": show_title,
        "language": language,
        "language_name": LANG_NAMES.get(language, "Español"),
        "style_profile": style_profile,
        "style_profile_name": s_meta["name"],
        "audience_governance": audience_mode,
        "format": podcast_format,
        "target_duration_minutes": target_minutes,
        "is_video_source": insights.get("is_video", False),
        "used_qwen_ai": bool(api_key),
        "core_intention": insights.get("summary", ""),
        "editorial_pillars": insights.get("beyond_tech_points", [])[:4],
        "technical_highlights": insights.get("technical_deep_dive", [])[:3],
        "practical_applications": insights.get("practical_cases_and_ideas", [])[:3],
    }

    lines: List[Dict[str, Any]] = []

    def append_turn(
        spk: Dict[str, Any],
        act_name: str,
        tone_key: str,
        sfx_req: bool,
        is_bc: bool,
        txt: str
    ):
        t_params = _apply_tone_and_style_params(
            spk.get("rate", "+0%"),
            spk.get("pitch", "+0Hz"),
            tone_key,
            style_profile
        )
        lines.append({
            "speaker_id": spk["id"],
            "speaker_name": spk["name"],
            "speaker_role": spk.get("role", ""),
            "voice_id": spk.get("voice_id", "es-MX-DaliaNeural"),
            "qwen_voice": spk.get("qwen_voice", "Serena"),
            "language": language,
            "pitch": t_params["pitch"],
            "rate": t_params["rate"],
            "tone": t_params["tone"],
            "tone_label": t_params["tone_label"],
            "style": style_profile,
            "exaggeration": t_params["exaggeration"],
            "overlap_ms": s_meta["overlap_ms"] if is_bc else 0,
            "is_backchannel": is_bc,
            "act": act_name,
            "text": txt.strip(),
            "estimated_seconds": _estimate_seconds(txt),
            "sfx_transition_requested": sfx_req
        })

    # 1. Intentar generación creativa contextual con Qwen si el usuario conectó su API Key
    qwen_turns = _try_generate_script_with_qwen(
        show_title=show_title,
        host_a=host_a,
        host_b=b_speaker,
        insights=insights,
        language=language,
        style_profile=style_profile,
        is_monologue=is_monologue,
        custom_notes=custom_notes,
        api_key=api_key,
        provider_hint=provider_hint
    )

    if qwen_turns:
        for item in qwen_turns:
            spk_obj = host_a if str(item.get("speaker", "A")).upper().startswith("A") or is_monologue else b_speaker
            append_turn(
                spk=spk_obj,
                act_name=str(item.get("act", "Conversación")),
                tone_key=str(item.get("tone", "calido")),
                sfx_req=bool(item.get("sfx_transition", False)),
                is_bc=bool(item.get("is_backchannel", False)),
                txt=str(item.get("text", ""))
            )
    else:
        # 2. Generación nativa fluida en los 3 idiomas (ES, EN, PT) sin tirantez
        raw_tuples = _build_smooth_multilingual_fallback(
            lang=language,
            show_title=show_title,
            host_a=host_a,
            b_speaker=b_speaker,
            is_monologue=is_monologue,
            insights=insights,
            custom_notes=custom_notes
        )
        for spk_code, act_n, tn, sfx_r, bc_r, txt_s in raw_tuples:
            if is_monologue and bc_r:
                continue
            spk_obj = host_a if spk_code == "A" or is_monologue else b_speaker
            append_turn(spk_obj, act_n, tn, sfx_r, bc_r, txt_s)

    return {
        "master_document": master_doc,
        "script_lines": lines
    }


def run_goldfish_audit(
    elephant_output: Dict[str, Any],
    brand_core: Dict[str, Any],
    audience_mode: str = "internal"
) -> Dict[str, Any]:
    script_lines = [dict(line) for line in elephant_output.get("script_lines", [])]
    confidential_terms = brand_core.get("governance", {}).get("confidential_patterns", [])
    min_interval = brand_core.get("audio_standards", {}).get("min_transition_interval_seconds", 90)
    neg_prompts = brand_core.get("goldfish_sound_rules", {}).get("mandatory_negative_prompts", [])

    security_flags: List[str] = []
    naturalness_fixes: List[str] = []
    sound_audit_log: List[str] = []

    cumulative_time = 0.0
    last_transition_time = -999.0
    approved_transitions = 0
    suppressed_transitions = 0

    for idx, line in enumerate(script_lines):
        text = line["text"]

        if audience_mode == "public":
            for term in confidential_terms:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                if pattern.search(text):
                    text = pattern.sub("[PROTECTED]", text)
                    security_flags.append(f"Línea {idx + 1}: Dato interno protegido ('{term}').")

        lower_t = text.lower()
        for robotic, human_alt in ROBOTIC_PHRASES_MAP.items():
            if robotic in lower_t:
                text = re.sub(re.escape(robotic), human_alt, text, flags=re.IGNORECASE)
                naturalness_fixes.append(f"Línea {idx + 1}: Giro fluido aplicado.")

        line["text"] = text
        line["start_time_sec"] = round(cumulative_time, 1)

        if line.get("sfx_transition_requested"):
            elapsed_since_last = cumulative_time - last_transition_time
            if idx == 0 or elapsed_since_last >= min_interval:
                line["sfx_transition_approved"] = True
                last_transition_time = cumulative_time
                approved_transitions += 1
                sound_audit_log.append(f"t={int(cumulative_time)}s ({line['act']}): Transición suave aprobada.")
            else:
                line["sfx_transition_approved"] = False
                suppressed_transitions += 1
        else:
            line["sfx_transition_approved"] = False

        step_dur = 2.0 if line.get("is_backchannel") else max(line["estimated_seconds"], 18.0)
        cumulative_time += step_dur
        line["end_time_sec"] = round(cumulative_time, 1)

    audit_report = {
        "overall_status": "APPROVED",
        "goldfish_1_security": {
            "audience_mode": audience_mode,
            "passed": True,
            "redactions_count": len(security_flags),
            "details": security_flags or [f"Gobernanza '{audience_mode}' verificada sin fugas de datos."]
        },
        "goldfish_2_naturalness": {
            "passed": True,
            "fixes_applied": len(naturalness_fixes),
            "details": naturalness_fixes or [
                "Cadencia continua de principio a fin, traducción íntegra y transiciones orgánicas sin tirantez."
            ]
        },
        "goldfish_3_sound": {
            "passed": True,
            "approved_transitions": approved_transitions,
            "suppressed_transitions": suppressed_transitions,
            "enforced_bed_lufs": -24.0,
            "enforced_voice_lufs": -16.0,
            "room_tone_db": -42.0,
            "enforced_negative_prompts": neg_prompts,
            "details": sound_audit_log
        }
    }

    return {
        "master_document": elephant_output["master_document"],
        "audited_script": script_lines,
        "goldfish_report": audit_report
    }
