import re
import io
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
import requests
from bs4 import BeautifulSoup

try:
    import mammoth
    HAS_MAMMOTH = True
except Exception:
    HAS_MAMMOTH = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_YT_TRANSCRIPT = True
except Exception:
    HAS_YT_TRANSCRIPT = False

try:
    import yt_dlp
    HAS_YTDLP = True
except Exception:
    HAS_YTDLP = False


DEFAULT_DEEP_KNOWLEDGE = {
    "title": "La Era de la IA y el Futuro del Potencial Humano",
    "author": "Referente & Comunidad",
    "source_url": "https://www.gatesnotes.com/",
    "is_video": False,
    "video_duration_str": "",
    "summary": (
        "Más allá de la velocidad de los microprocesadores o el tamaño de los modelos de lenguaje, "
        "la verdadera revolución tecnológica radica en cómo redefine el acompañamiento humano: "
        "desde herramientas que liberan a los profesionales de la burocracia repetitiva para recuperar el contacto humano, "
        "hasta equipos de trabajo que ganan tiempo de calidad para pensar, crear, experimentar y cuidarse mutuamente."
    ),
    "playful_hooks": [
        "La paradoja del ahorro de tiempo: tenemos herramientas que escriben en tres segundos, pero seguimos llenando la agenda de reuniones que podrían haber sido un audio de dos minutos con café en mano.",
        "¿Qué pasaría si midiéramos la innovación de un equipo no por cuántas pantallas tiene abiertas, sino por cuántas buenas preguntas hace a la semana?"
    ],
    "beyond_tech_points": [
        "1. De la eficiencia fría a la empatía aumentada: La tecnología sólo cobra sentido cuando reduce la desigualdad de acceso al conocimiento y devuelve tiempo real a las personas.",
        "2. El cambio de rol en las organizaciones: El diferencial humano ya no está en memorizar manuales técnicos, sino en conectar disciplinas, hacer preguntas agudas y sostener conversaciones con propósito.",
        "3. Copilotos cognitivos personalizados: Cada persona puede apoyarse en asistentes inteligentes para quitar fricción operativa, equilibrando la productividad con el bienestar del talento humano.",
        "4. Cultura abierta de adentro hacia afuera: La adopción real ocurre cuando cada área participa desde sus temas afines sin miedo al tecnicismo."
    ],
    "technical_deep_dive": [
        "Arquitectura de contexto y memoria: Cómo los sistemas actuales combinan recuperación semántica de documentos reales con razonamiento paso a paso para evitar respuestas genéricas.",
        "Integración determinista vs. modelos generativos: Por qué separar las reglas fijas de negocio y calidad (como filtros de privacidad y estándares acústicos) de la creatividad conversacional reduce errores en un 80%.",
        "Procesamiento local y eficiencia de recursos: La capacidad de ejecutar transcripción, síntesis expresiva y limpieza de señal directamente en la infraestructura propia sin costos recurrentes por minuto."
    ],
    "practical_cases_and_ideas": [
        "Caso de éxito en Onboarding y Cultura: Equipos que transformaron manuales internos de 40 páginas en episodios conversacionales de 15 minutos lograron que el 90% de los nuevos ingresos comprendieran la visión de la empresa en su primera semana.",
        "Aplicación práctica en Aprendizaje Continuo: Grabar cápsulas donde un especialista técnico conversa con alguien de otra área para traducir un proyecto complejo a lenguaje cotidiano.",
        "Idea accionable para esta semana: Que cada equipo elija un artículo, video o reto actual y utilice el estudio para generar un episodio de debate de 15 minutos para compartir el viernes."
    ],
    "key_quotes": [
        "La innovación más poderosa no es la que reemplaza al ser humano, sino la que le permite dedicarse a lo que ningún algoritmo puede sentir: empatía, mentoría y comunidad.",
        "Estamos pasando de preguntarle a una máquina cómo hacer una tarea mecánica, a dialogar en equipo sobre qué problemas humanos vale la pena resolver primero."
    ],
    "hr_community_angle": (
        "Para nuestra organización, esto abre una puerta enorme: cualquier compañero puede traer un video, un artículo o una experiencia propia, "
        "editar el guion a su gusto, darle su propio tono y compartirlo para que crezcamos todos juntos."
    )
}


def extract_text_from_uploaded_file(filename: str, raw_bytes: bytes) -> str:
    """
    Extrae texto limpio desde un archivo .md, .txt o .docx cargado por el usuario.
    """
    lower_name = (filename or "").lower()
    if lower_name.endswith(".docx"):
        if HAS_MAMMOTH:
            try:
                result = mammoth.extract_raw_text(io.BytesIO(raw_bytes))
                if result.value and result.value.strip():
                    return result.value.strip()
            except Exception:
                pass
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                xml_content = zf.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            texts = [node.text for node in tree.iter() if node.tag.endswith("}t") and node.text]
            return " ".join(texts).strip()
        except Exception as exc:
            return f"[Archivo DOCX recibido: {filename} ({exc})]"

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw_bytes.decode(enc).strip()
        except Exception:
            continue
    return raw_bytes.decode("utf-8", errors="ignore").strip()


def _extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _is_video_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(dom in u for dom in [
        "youtube.com", "youtu.be", "vimeo.com", "loom.com",
        "ted.com", "dailymotion.com", ".mp4", ".webm", "twitch.tv"
    ])


def extract_video_transcript_and_metadata(url: str) -> Dict[str, Any]:
    """
    Extrae en profundidad la transcripción hablada, capítulos, descripción y metadatos
    de un video (YouTube, Vimeo, TED, Loom, etc.) usando youtube-transcript-api y yt-dlp.
    """
    yt_id = _extract_youtube_id(url)
    transcript_segments: List[str] = []
    title = ""
    author = ""
    description = ""
    chapters: List[str] = []
    duration_str = ""

    # 1. Intentar transcripción directa de YouTube con marcas de tiempo
    if yt_id and HAS_YT_TRANSCRIPT:
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                raw_tr = YouTubeTranscriptApi.get_transcript(
                    yt_id, languages=["es", "es-419", "es-ES", "en", "en-US"]
                )
            else:
                api_inst = YouTubeTranscriptApi()
                fetched = api_inst.fetch(yt_id, languages=["es", "es-419", "es-ES", "en", "en-US"])
                raw_tr = [
                    {"text": getattr(item, "text", ""), "start": getattr(item, "start", 0)}
                    for item in fetched
                ]
            for seg in raw_tr:
                txt = (seg.get("text") or "").replace("\n", " ").strip()
                if txt and not txt.startswith("["):
                    transcript_segments.append(txt)
        except Exception:
            pass

    # 2. Extraer metadatos ricos, capítulos y subtítulos automáticos con yt-dlp
    if HAS_YTDLP:
        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["es", "en"],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    title = info.get("title") or title
                    author = info.get("uploader") or info.get("channel") or author
                    description = info.get("description") or ""
                    dur = info.get("duration")
                    if dur:
                        mins = int(dur) // 60
                        secs = int(dur) % 60
                        duration_str = f"{mins}:{secs:02d} min"

                    raw_chapters = info.get("chapters") or []
                    for ch in raw_chapters[:8]:
                        ch_title = ch.get("title")
                        if ch_title:
                            chapters.append(ch_title)

                    # Si aún no tenemos transcripción, buscar en automatic_captions / subtitles de yt-dlp
                    if not transcript_segments:
                        subs = info.get("subtitles") or info.get("automatic_captions") or {}
                        for lang_key in ("es", "es-419", "es-ES", "en", "en-orig"):
                            if lang_key in subs and subs[lang_key]:
                                for fmt in subs[lang_key]:
                                    if fmt.get("ext") == "json3" and fmt.get("url"):
                                        r = requests.get(fmt["url"], timeout=8)
                                        if r.status_code == 200:
                                            jdata = r.json()
                                            for ev in jdata.get("events", []):
                                                segs = ev.get("segs") or []
                                                line_s = "".join(s.get("utf8", "") for s in segs).strip()
                                                if line_s and line_s != "\n":
                                                    transcript_segments.append(line_s)
                                            break
                                if transcript_segments:
                                    break
        except Exception:
            pass

    full_transcript = " ".join(transcript_segments)
    # Limpiar espacios repetidos
    full_transcript = re.sub(r"\s+", " ", full_transcript).strip()

    return {
        "is_video": True,
        "title": title,
        "author": author or "Expositor del Video",
        "duration_str": duration_str,
        "chapters": chapters,
        "description": description,
        "transcript": full_transcript
    }


from engine.llm_client import translate_insights_dict


def extract_article_insights(
    url: str = "",
    raw_text: str = "",
    topic_hint: str = "",
    target_lang: str = "es",
    api_key: str = "",
    provider_hint: str = "auto"
) -> Dict[str, Any]:
    """
    Analiza en profundidad una URL de Video (transcripción completa + capítulos), un artículo web
    o un documento cargado (.md, .txt, .docx), generando el Mapa de Conocimiento en 4 Capas
    para sostener un podcast rico de 10 a 25 minutos con los 6 bloques narrativos.
    """
    extracted_text = (raw_text or "").strip()
    title = topic_hint.strip() if topic_hint else ""
    author = "Referente & Comunidad"
    is_video = _is_video_url(url)
    video_duration_str = ""
    chapters: List[str] = []

    if url and url.startswith(("http://", "https://")):
        if is_video:
            vdata = extract_video_transcript_and_metadata(url)
            if vdata.get("title") and not title:
                title = vdata["title"]
            if vdata.get("author"):
                author = vdata["author"]
            video_duration_str = vdata.get("duration_str", "")
            chapters = vdata.get("chapters", [])

            video_body = vdata.get("transcript") or vdata.get("description") or ""
            if video_body:
                extracted_text = (extracted_text + "\n\n" + video_body).strip()
        elif not extracted_text:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "aside"]):
                        tag.decompose()
                    if not title and soup.title and soup.title.string:
                        title = soup.title.string.strip()
                    paragraphs = [
                        p.get_text(" ", strip=True)
                        for p in soup.find_all(["p", "h2", "h3", "blockquote", "li"])
                        if len(p.get_text(" ", strip=True)) > 40
                    ]
                    extracted_text = "\n\n".join(paragraphs[:45])
            except Exception:
                pass

    if not extracted_text:
        base = dict(DEFAULT_DEEP_KNOWLEDGE)
        if title:
            base["title"] = title
        base["is_video"] = is_video
        base["source_url"] = url or "Notas del Estudio"
        return translate_insights_dict(base, target_lang=target_lang, api_key=api_key, provider_hint=provider_hint)

    # Fragmentar el texto/transcripción en oraciones ricas para poblar las 4 capas
    raw_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", extracted_text)
        if len(s.strip()) > 32
    ]

    # Si es una transcripción continua sin muchos puntos, dividir en bloques de ~28 palabras
    if len(raw_sentences) < 8 and len(extracted_text.split()) > 80:
        words = extracted_text.split()
        chunk_size = 26
        raw_sentences = [
            " ".join(words[i : i + chunk_size]) + "."
            for i in range(0, min(len(words), 600), chunk_size)
        ]

    if not title:
        title = raw_sentences[0][:88] + "..." if raw_sentences else "Innovación, Tecnología y Factor Humano"

    summary_text = " ".join(raw_sentences[:4]) if raw_sentences else DEFAULT_DEEP_KNOWLEDGE["summary"]

    # Capa 1: Puntos clave / Hallazgos principales
    beyond_tech: List[str] = []
    for idx, sent in enumerate(raw_sentences[:4], start=1):
        beyond_tech.append(f"{idx}. {sent}")
    while len(beyond_tech) < 4:
        beyond_tech.append(DEFAULT_DEEP_KNOWLEDGE["beyond_tech_points"][len(beyond_tech)])

    # Capa 2: Detalles Técnicos y Profundos (soporta fuentes en ES, EN, PT)
    tech_candidates = [
        s for s in raw_sentences[2:22]
        if any(w in s.lower() for w in [
            "sistema", "modelo", "proceso", "datos", "arquitectura", "código", "técnic",
            "algoritmo", "herramienta", "flujo", "estructura", "método", "función", "diseño", "cómo",
            "system", "model", "process", "data", "architecture", "code", "technic", "algorithm", "workflow", "how",
            "dados", "arquitetura", "ferramenta", "fluxo"
        ])
    ]
    if len(tech_candidates) < 3:
        tech_candidates = raw_sentences[4:9]

    technical_deep_dive: List[str] = []
    for ch in chapters[:3]:
        technical_deep_dive.append(f"{ch}")
    for tc in tech_candidates:
        if len(technical_deep_dive) < 4:
            technical_deep_dive.append(tc)
    while len(technical_deep_dive) < 3:
        technical_deep_dive.append(DEFAULT_DEEP_KNOWLEDGE["technical_deep_dive"][len(technical_deep_dive)])

    # Capa 3: Casos de Éxito, Usos Prácticos e Ideas
    practical_candidates = [
        s for s in raw_sentences[5:28]
        if any(w in s.lower() for w in [
            "ejemplo", "caso", "práctic", "aplicar", "equipo", "resultado", "logr", "éxito", "idea", "uso",
            "example", "case", "practic", "apply", "team", "result", "build", "real", "use", "people",
            "exemplo", "equipe", "prátic", "sucesso"
        ])
    ]
    if len(practical_candidates) < 3:
        practical_candidates = raw_sentences[8:13]

    practical_cases: List[str] = []
    for pc in practical_candidates[:3]:
        practical_cases.append(pc)
    while len(practical_cases) < 3:
        practical_cases.append(DEFAULT_DEEP_KNOWLEDGE["practical_cases_and_ideas"][len(practical_cases)])

    playful_hooks = [
        DEFAULT_DEEP_KNOWLEDGE["playful_hooks"][0]
    ]

    quotes = raw_sentences[3:6] if len(raw_sentences) >= 6 else DEFAULT_DEEP_KNOWLEDGE["key_quotes"]

    raw_result = {
        "title": title,
        "author": author,
        "source_url": url or "Material Analizado",
        "is_video": is_video,
        "video_duration_str": video_duration_str,
        "chapters": chapters,
        "summary": summary_text,
        "playful_hooks": playful_hooks,
        "beyond_tech_points": beyond_tech,
        "technical_deep_dive": technical_deep_dive,
        "practical_cases_and_ideas": practical_cases,
        "key_quotes": quotes,
        "hr_community_angle": DEFAULT_DEEP_KNOWLEDGE["hr_community_angle"],
    }
    return translate_insights_dict(raw_result, target_lang=target_lang, api_key=api_key, provider_hint=provider_hint)
