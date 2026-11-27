import os
import re
import json
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.ingestor import extract_article_insights, extract_text_from_uploaded_file
from engine.elephant_goldfish import run_elephant_phase, run_goldfish_audit
from engine.llm_client import call_qwen_chat, LANG_NAMES
from engine.auth_vault import (
    decode_google_jwt_payload,
    login_or_register_gmail_user,
    get_user_by_session,
    update_user_qwen_settings,
    logout_session,
    _public_user_view
)
from engine.audio_dsp import (
    render_episode_master_audio,
    process_uploaded_voice_without_ai,
    process_human_and_ai_dialogue,
    generate_castopod_rss_xml
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Podcast Studio & Librería Podcast",
    description="Estudio Multilenguaje (ES/EN/PT) con Login Corporativo Gmail, Qwen3-TTS y Librería Estilo Spotify",
    version="4.0.0"
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def load_brand_core() -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, "brand_core.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_profiles_data() -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, "profiles.json")
    if not os.path.exists(path):
        return {"languages": [], "profiles": [], "voices_by_language": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_shows_data() -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, "shows.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_shows_data(data: Dict[str, Any]) -> None:
    path = os.path.join(CONFIG_DIR, "shows.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_library_data() -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, "library.json")
    if not os.path.exists(path):
        return {"playlists": [], "episodes": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data.setdefault("playlists", [])
        data.setdefault("episodes", [])
        return data


def save_library_data(data: Dict[str, Any]) -> None:
    path = os.path.join(CONFIG_DIR, "library.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class GmailLoginRequest(BaseModel):
    email: str = ""
    name: str = ""
    google_credential_jwt: str = ""


class QwenSettingsRequest(BaseModel):
    qwen_api_key: str = ""
    provider: str = "auto"
    use_qwen_tts: bool = True


class EpisodeRequest(BaseModel):
    show_id: str = "cafe-medialunas-max"
    language: str = "es"
    style_profile: str = "oficina"
    topic_title: str = ""
    source_url: str = ""
    raw_notes: str = ""
    podcast_format: str = "debate_2_hosts"
    audience_mode: str = "internal"
    target_minutes: int = 16
    participant_id: str = "persona_es_dalia"
    cohost_id: str = ""
    guest_name: str = ""
    use_voice_recording: bool = False
    use_preconfigured_agents: bool = True
    use_ai_dialogue_recording: bool = False
    synthesize_audio: bool = True


class ScriptRenderRequest(BaseModel):
    show_id: str = "cafe-medialunas-max"
    episode_title: str = "Episodio de Podcast"
    audience_mode: str = "internal"
    script_lines: List[Dict[str, Any]]


class ChatRefineRequest(BaseModel):
    instruction: str
    episode_title: str
    language: str = "es"
    show_id: str = "cafe-medialunas-max"
    script_lines: List[Dict[str, Any]]
    render_audio: bool = True


class PublishLibraryRequest(BaseModel):
    title: str
    show_id: str
    show_name: str
    language: str = "es"
    style_profile: str = "oficina"
    audience: str = "internal"
    status: str = "listo"
    playlists: List[str] = []
    format_label: str = "Conversación"
    cover_image: str
    audio_url: str
    description: str = ""
    waveform_bars: List[int] = []


class UpdateEpisodePlaylistsRequest(BaseModel):
    playlists: List[str] = []
    status: str = "listo"


@app.get("/")
async def serve_index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/api/auth/google-login")
async def auth_google_login(req: GmailLoginRequest):
    email = req.email.strip()
    name = req.name.strip()
    picture = ""

    if req.google_credential_jwt:
        decoded = decode_google_jwt_payload(req.google_credential_jwt)
        if decoded and decoded.get("email"):
            email = decoded["email"]
            name = decoded.get("name", name)
            picture = decoded.get("picture", "")

    try:
        res = login_or_register_gmail_user(CONFIG_DIR, email=email, name=name, picture=picture)
        return {"status": "ok", **res}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/auth/me")
async def auth_get_me(x_session_token: Optional[str] = Header(default="")):
    user = get_user_by_session(CONFIG_DIR, x_session_token or "")
    if not user:
        return {"status": "anonymous", "user": None}
    return {"status": "ok", "user": _public_user_view(user)}


@app.post("/api/auth/qwen-key")
async def auth_save_qwen_key(
    req: QwenSettingsRequest,
    x_session_token: Optional[str] = Header(default="")
):
    updated = update_user_qwen_settings(
        CONFIG_DIR,
        session_token=x_session_token or "",
        qwen_api_key=req.qwen_api_key,
        provider=req.provider,
        use_qwen_tts=req.use_qwen_tts
    )
    if not updated:
        raise HTTPException(status_code=401, detail="Inicia sesión con tu cuenta Gmail primero.")
    return {"status": "ok", "user": updated}


@app.post("/api/auth/logout")
async def auth_logout(x_session_token: Optional[str] = Header(default="")):
    logout_session(CONFIG_DIR, x_session_token or "")
    return {"status": "ok"}


@app.get("/api/config")
async def get_studio_config():
    shows_data = load_shows_data()
    profiles_data = load_profiles_data()
    lib_data = load_library_data()
    return {
        "brand_core": load_brand_core(),
        "google_client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "languages": profiles_data.get("languages", [
            {"id": "es", "name": "Español (Neutro Internacional)"},
            {"id": "en", "name": "English (International / US)"},
            {"id": "pt", "name": "Português (Brasil / Internacional)"}
        ]),
        "profiles": profiles_data.get("profiles", []),
        "voices_by_language": profiles_data.get("voices_by_language", {}),
        "shows": shows_data.get("shows", []),
        "personas": shows_data.get("personas", []),
        "preset_covers": shows_data.get("preset_covers", []),
        "playlists": lib_data.get("playlists", []),
        "library": lib_data.get("episodes", []),
        "formats": [
            {"id": "debate_2_hosts", "name": "Diálogo Entretenido de 2 Personajes (Conversación Humana)"},
            {"id": "entrevista_hibrida", "name": "Entrevista Abierta (Colaborador Invitado + Anfitrión)"},
            {"id": "monologo", "name": "Monólogo Reflexivo / Cápsula de Autor"}
        ]
    }


@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    extracted = extract_text_from_uploaded_file(file.filename, raw_bytes)
    first_line = next((l.strip("# ").strip() for l in extracted.splitlines() if len(l.strip()) > 4), "")
    return {
        "status": "ok",
        "filename": file.filename,
        "extracted_text": extracted,
        "suggested_topic": first_line[:110] if first_line else file.filename
    }


@app.post("/api/shows/update")
async def update_shows_list(payload: Dict[str, Any]):
    data = load_shows_data()
    if "shows" in payload and isinstance(payload["shows"], list):
        data["shows"] = payload["shows"]
    if "personas" in payload and isinstance(payload["personas"], list):
        data["personas"] = payload["personas"]
    if "preset_covers" in payload and isinstance(payload["preset_covers"], list):
        data["preset_covers"] = payload["preset_covers"]
    save_shows_data(data)
    return {
        "status": "ok",
        "shows": data["shows"],
        "personas": data["personas"],
        "preset_covers": data.get("preset_covers", [])
    }


@app.post("/api/playlists/update")
async def update_playlists_list(payload: Dict[str, Any]):
    lib_data = load_library_data()
    if "playlists" in payload and isinstance(payload["playlists"], list):
        lib_data["playlists"] = payload["playlists"]
    save_library_data(lib_data)
    return {"status": "ok", "playlists": lib_data["playlists"]}


@app.post("/api/generate-episode")
async def generate_episode(
    req: EpisodeRequest,
    x_session_token: Optional[str] = Header(default="")
):
    brand_core = load_brand_core()
    shows_data = load_shows_data()
    profiles_data = load_profiles_data()
    shows = shows_data.get("shows", [])

    # Obtener credenciales Qwen personales del usuario logueado con Gmail (si inició sesión)
    active_user = get_user_by_session(CONFIG_DIR, x_session_token or "")
    user_api_key = (active_user or {}).get("qwen_api_key", "")
    user_provider = (active_user or {}).get("provider", "auto")
    user_use_qwen_tts = bool((active_user or {}).get("use_qwen_tts", True))

    lang = req.language if req.language in ("es", "en", "pt") else "es"
    lang_voices = profiles_data.get("voices_by_language", {}).get(lang, [])
    all_personas = lang_voices + shows_data.get("personas", [])

    selected_show = next((s for s in shows if s["id"] == req.show_id), shows[0])
    chosen_a = next((p for p in all_personas if p["id"] == req.participant_id), None)
    chosen_b = next((p for p in all_personas if p["id"] == req.cohost_id), None)

    default_host_a = lang_voices[0] if len(lang_voices) > 0 else all_personas[0]
    default_host_b = lang_voices[1] if len(lang_voices) > 1 else all_personas[-1]

    # Si la voz elegida pertenece a otro idioma, adaptar automáticamente a la voz equivalente del idioma seleccionado
    if chosen_a and chosen_a.get("lang") and chosen_a.get("lang") != lang and lang_voices:
        chosen_a = default_host_a

    host_1 = chosen_a or default_host_a
    host_2 = chosen_b or (default_host_b if host_1["id"] != default_host_b["id"] else default_host_a)
    if host_1["id"] == host_2["id"] and len(lang_voices) > 1:
        host_2 = next((p for p in lang_voices if p["id"] != host_1["id"]), default_host_b)

    show_with_hosts = dict(selected_show)
    show_with_hosts["hosts"] = [host_1, host_2]

    default_titles = {
        "es": "Innovación con propósito humano y participación abierta",
        "en": "Human-centric innovation and open community collaboration",
        "pt": "Inovação com propósito humano e colaboração aberta"
    }
    topic_hint = req.topic_title.strip() or default_titles.get(lang, default_titles["es"])

    # 1. Extraer y TRADUCIR ÍNTEGRAMENTE al idioma seleccionado (es, en, pt)
    insights = extract_article_insights(
        url=req.source_url,
        raw_text=req.raw_notes,
        topic_hint=topic_hint,
        target_lang=lang,
        api_key=user_api_key,
        provider_hint=user_provider
    )

    # 2. Generar guion fluido en 6 bloques en el idioma elegido (con Qwen si hay API Key)
    elephant_data = run_elephant_phase(
        show=show_with_hosts,
        brand_core=brand_core,
        insights=insights,
        podcast_format=req.podcast_format,
        audience_mode=req.audience_mode,
        target_minutes=req.target_minutes,
        guest_name=req.guest_name,
        custom_notes=req.raw_notes,
        style_profile=req.style_profile,
        language=lang,
        api_key=user_api_key,
        provider_hint=user_provider
    )

    audited_pack = run_goldfish_audit(
        elephant_output=elephant_data,
        brand_core=brand_core,
        audience_mode=req.audience_mode
    )

    slug_base = re.sub(r"[^a-z0-9]+", "_", insights.get("title", "episodio").lower()).strip("_")[:30] or "episodio"
    slug = f"{slug_base}_{lang}_{int(time.time())}"

    audio_result = await render_episode_master_audio(
        audited_script=audited_pack["audited_script"],
        output_dir=OUTPUT_DIR,
        episode_slug=slug,
        bed_style=selected_show.get("music_bed_style", "pastel_acoustic_warm"),
        use_tts=req.synthesize_audio,
        qwen_api_key=user_api_key,
        use_qwen_tts=user_use_qwen_tts
    )

    rss_xml = generate_castopod_rss_xml(
        episode_meta=audited_pack["master_document"],
        audio_url=f"/outputs/{audio_result['wav_filename']}"
    )
    rss_path = os.path.join(OUTPUT_DIR, audio_result["rss_filename"])
    with open(rss_path, "w", encoding="utf-8") as f:
        f.write(rss_xml)

    return {
        "status": "ok",
        "show": show_with_hosts,
        "insights": insights,
        "master_document": audited_pack["master_document"],
        "goldfish_report": audited_pack["goldfish_report"],
        "script": audio_result["timed_segments"],
        "media": {
            "audio_url": f"/outputs/{audio_result['wav_filename']}",
            "srt_url": f"/outputs/{audio_result['srt_filename']}",
            "rss_url": f"/outputs/{audio_result['rss_filename']}",
            "duration_seconds": audio_result["duration_seconds"],
            "waveform_bars": audio_result["waveform_bars_62"]
        }
    }


@app.post("/api/render-edited-script")
async def render_edited_script(
    req: ScriptRenderRequest,
    x_session_token: Optional[str] = Header(default="")
):
    active_user = get_user_by_session(CONFIG_DIR, x_session_token or "")
    user_api_key = (active_user or {}).get("qwen_api_key", "")
    user_use_qwen_tts = bool((active_user or {}).get("use_qwen_tts", True))

    shows = load_shows_data().get("shows", [])
    selected_show = next((s for s in shows if s["id"] == req.show_id), shows[0])
    slug_base = re.sub(r"[^a-z0-9]+", "_", req.episode_title.lower()).strip("_")[:28] or "guion_editado"
    slug = f"{slug_base}_{int(time.time())}"

    audio_result = await render_episode_master_audio(
        audited_script=req.script_lines,
        output_dir=OUTPUT_DIR,
        episode_slug=slug,
        bed_style=selected_show.get("music_bed_style", "pastel_acoustic_warm"),
        use_tts=True,
        qwen_api_key=user_api_key,
        use_qwen_tts=user_use_qwen_tts
    )
    return {
        "status": "ok",
        "script": audio_result["timed_segments"],
        "media": {
            "audio_url": f"/outputs/{audio_result['wav_filename']}",
            "srt_url": f"/outputs/{audio_result['srt_filename']}",
            "duration_seconds": audio_result["duration_seconds"],
            "waveform_bars": audio_result["waveform_bars_62"]
        }
    }


@app.post("/api/dsp-clean-voice")
async def dsp_clean_voice(
    audio_file: UploadFile = File(...),
    add_bed: bool = Form(True),
    bed_style: str = Form("pastel_acoustic_warm"),
    dialogue_with_ai: bool = Form(False),
    participant_id: str = Form("persona_es_jorge"),
    script_json: str = Form("[]"),
    x_session_token: Optional[str] = Header(default="")
):
    active_user = get_user_by_session(CONFIG_DIR, x_session_token or "")
    user_api_key = (active_user or {}).get("qwen_api_key", "")
    user_use_qwen_tts = bool((active_user or {}).get("use_qwen_tts", True))

    raw_bytes = await audio_file.read()
    slug = f"grabacion_{int(time.time())}"
    shows_data = load_shows_data()
    profiles_data = load_profiles_data()
    all_voices = []
    for vlist in profiles_data.get("voices_by_language", {}).values():
        all_voices.extend(vlist)
    all_voices.extend(shows_data.get("personas", []))
    ai_persona = next((p for p in all_voices if p["id"] == participant_id), all_voices[0] if all_voices else {})

    try:
        script_lines = json.loads(script_json) if script_json else []
    except Exception:
        script_lines = []

    if dialogue_with_ai:
        res = await process_human_and_ai_dialogue(
            raw_bytes=raw_bytes,
            script_lines=script_lines,
            ai_persona=ai_persona,
            output_dir=OUTPUT_DIR,
            slug=slug,
            bed_style=bed_style,
            qwen_api_key=user_api_key,
            use_qwen_tts=user_use_qwen_tts
        )
        return {
            "status": "ok",
            "mode": "dialogue_with_ai",
            "audio_url": f"/outputs/{res['wav_filename']}",
            "details": res
        }

    res = process_uploaded_voice_without_ai(
        raw_bytes=raw_bytes,
        output_dir=OUTPUT_DIR,
        slug=slug,
        add_ambient_bed=add_bed,
        bed_style=bed_style
    )
    return {
        "status": "ok",
        "mode": "voice_master",
        "audio_url": f"/outputs/{res['wav_filename']}",
        "details": res
    }


@app.post("/api/chat-refine")
async def chat_refine_podcast(
    req: ChatRefineRequest,
    x_session_token: Optional[str] = Header(default="")
):
    active_user = get_user_by_session(CONFIG_DIR, x_session_token or "")
    user_api_key = (active_user or {}).get("qwen_api_key", "")
    user_provider = (active_user or {}).get("provider", "auto")
    user_use_qwen_tts = bool((active_user or {}).get("use_qwen_tts", True))

    instr = req.instruction.strip()
    updated_lines = [dict(l) for l in req.script_lines]
    lang_label = LANG_NAMES.get(req.language, "Español")
    used_qwen = False

    # Si el usuario tiene API Key de Qwen, Qwen comprende todo el guion y lo edita contextualmente
    if user_api_key and updated_lines:
        prompt = (
            f"Eres un editor experto de podcasts en {lang_label}. "
            f"Aplica la siguiente instrucción del usuario sobre el guion manteniendo todos los metadatos y la fluidez humana: «{instr}».\n"
            f"Devuelve EXCLUSIVAMENTE el array JSON completo de líneas manteniendo la clave 'text' editada donde corresponda (misma longitud o intervenciones ajustadas):\n"
            + json.dumps(
                [{"idx": i, "speaker_name": l.get("speaker_name"), "act": l.get("act"), "text": l.get("text")} for i, l in enumerate(updated_lines)],
                ensure_ascii=False
            )
        )
        q_resp = call_qwen_chat(
            [{"role": "user", "content": prompt}],
            api_key=user_api_key,
            provider_hint=user_provider,
            temperature=0.55,
            max_tokens=3200
        )
        if q_resp:
            try:
                c_json = q_resp.strip()
                if "```" in c_json:
                    c_json = c_json.split("```")[1]
                    if c_json.startswith("json"):
                        c_json = c_json[4:]
                edited_arr = json.loads(c_json.strip())
                if isinstance(edited_arr, list) and len(edited_arr) == len(updated_lines):
                    for idx_i, item in enumerate(edited_arr):
                        if item.get("text"):
                            updated_lines[idx_i]["text"] = str(item["text"]).strip()
                    used_qwen = True
            except Exception:
                pass

    if not used_qwen and updated_lines:
        if any(w in instr.lower() for w in ["inicio", "intro", "apertura", "bienvenida", "opening"]):
            updated_lines[0]["text"] = f"{updated_lines[0]['text']} {instr}."
        elif any(w in instr.lower() for w in ["cierre", "final", "despedida", "closing"]):
            updated_lines[-1]["text"] = f"{updated_lines[-1]['text']} {instr}."
        else:
            mid = len(updated_lines) // 2
            speaker_ref = dict(updated_lines[mid])
            speaker_ref["text"] = instr
            speaker_ref["sfx_transition_approved"] = False
            updated_lines.insert(mid + 1, speaker_ref)

    media_info = None
    if req.render_audio and updated_lines:
        shows = load_shows_data().get("shows", [])
        selected_show = next((s for s in shows if s["id"] == req.show_id), shows[0])
        slug = f"refinado_{req.language}_{int(time.time())}"
        audio_res = await render_episode_master_audio(
            audited_script=updated_lines,
            output_dir=OUTPUT_DIR,
            episode_slug=slug,
            bed_style=selected_show.get("music_bed_style", "pastel_acoustic_warm"),
            use_tts=True,
            qwen_api_key=user_api_key,
            use_qwen_tts=user_use_qwen_tts
        )
        updated_lines = audio_res["timed_segments"]
        media_info = {
            "audio_url": f"/outputs/{audio_res['wav_filename']}",
            "srt_url": f"/outputs/{audio_res['srt_filename']}",
            "duration_seconds": audio_res["duration_seconds"],
            "waveform_bars": audio_res["waveform_bars_62"]
        }

    engine_tag = "Qwen AI" if used_qwen else "Editor del Estudio"
    return {
        "status": "ok",
        "reply": f"{engine_tag}: ajuste aplicado y audio actualizado («{instr}»).",
        "script": updated_lines,
        "media": media_info
    }


@app.post("/api/upload-cover")
async def upload_cover_image(image_file: UploadFile = File(...)):
    ext = os.path.splitext(image_file.filename or "cover.png")[1].lower() or ".png"
    out_name = f"cover_{int(time.time())}{ext}"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    raw = await image_file.read()
    with open(out_path, "wb") as f:
        f.write(raw)
    return {"status": "ok", "cover_url": f"/outputs/{out_name}"}


@app.get("/api/library")
async def get_library_episodes():
    lib_data = load_library_data()
    return {
        "status": "ok",
        "playlists": lib_data.get("playlists", []),
        "episodes": lib_data.get("episodes", [])
    }


@app.post("/api/library/publish")
async def publish_to_library(
    req: PublishLibraryRequest,
    x_session_token: Optional[str] = Header(default="")
):
    active_user = get_user_by_session(CONFIG_DIR, x_session_token or "")
    author_name = (active_user or {}).get("name", "Comunidad")

    lib_data = load_library_data()
    episodes = lib_data.get("episodes", [])
    new_ep = {
        "id": f"ep_{int(time.time())}",
        "title": req.title,
        "show_id": req.show_id,
        "show_name": req.show_name,
        "language": req.language,
        "author_name": author_name,
        "style_profile": req.style_profile,
        "date_label": f"Hoy · {req.show_name} ({req.language.upper()}) · {author_name}",
        "audience": req.audience,
        "status": req.status,
        "playlists": req.playlists,
        "format_label": req.format_label,
        "cover_image": req.cover_image,
        "audio_url": req.audio_url,
        "duration_label": "15 min",
        "description": req.description or "Episodio creado en Podcast Studio.",
        "waveform_bars": req.waveform_bars or [10, 14, 18, 24, 30, 26, 18, 12, 15, 22, 28, 34, 29, 21, 15, 11, 14, 19, 25, 21, 16, 12, 10, 14, 20, 27, 23, 17, 13, 11, 15, 21, 26, 22, 17, 13, 10, 12, 17, 23, 28, 24, 18, 13, 10, 12, 16, 20, 17, 13, 10, 9, 11, 13, 12, 10, 8, 9, 7, 8, 7, 6]
    }
    episodes.insert(0, new_ep)
    lib_data["episodes"] = episodes
    save_library_data(lib_data)
    return {"status": "ok", "episode": new_ep, "episodes": episodes}


@app.post("/api/library/{episode_id}/playlists")
async def update_episode_playlists(episode_id: str, req: UpdateEpisodePlaylistsRequest):
    lib_data = load_library_data()
    for ep in lib_data.get("episodes", []):
        if ep["id"] == episode_id:
            ep["playlists"] = req.playlists
            ep["status"] = req.status
            break
    save_library_data(lib_data)
    return {"status": "ok", "episodes": lib_data["episodes"]}


@app.delete("/api/library/{episode_id}")
async def delete_episode_from_library(episode_id: str):
    lib_data = load_library_data()
    lib_data["episodes"] = [ep for ep in lib_data.get("episodes", []) if ep["id"] != episode_id]
    save_library_data(lib_data)
    return {"status": "ok", "episodes": lib_data["episodes"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8085, reload=False)
