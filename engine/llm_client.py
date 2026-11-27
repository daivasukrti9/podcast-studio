import os
import json
import urllib.parse
from typing import Dict, Any, List, Optional
import requests


LANG_NAMES = {
    "es": "Español Latinoamericano Neutro",
    "en": "Natural International English",
    "pt": "Português Brasileiro Natural"
}


def resolve_qwen_endpoint(api_key: str, provider_hint: str = "auto") -> Optional[Dict[str, str]]:
    """
    Detecta automáticamente el endpoint de Qwen según el formato de la API Key del usuario
    o el proveedor seleccionado:
    - Groq (`gsk_...`): https://api.groq.com/openai/v1/chat/completions -> qwen/qwen3-32b
    - OpenRouter (`sk-or-...`): https://openrouter.ai/api/v1/chat/completions -> qwen/qwen3-235b-a22b:free
    - Alibaba Model Studio Internacional (`sk-...`): https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions -> qwen-plus
    """
    key = (api_key or os.environ.get("QWEN_API_KEY", "")).strip()
    if not key:
        return None

    prov = (provider_hint or "auto").lower()
    if prov == "groq" or key.startswith("gsk_"):
        return {
            "provider": "groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "model": "qwen/qwen3-32b",
            "api_key": key
        }
    if prov == "openrouter" or key.startswith("sk-or-"):
        return {
            "provider": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": "qwen/qwen3-235b-a22b:free",
            "fallback_model": "openrouter/free",
            "api_key": key
        }
    return {
        "provider": "dashscope",
        "url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": "qwen-plus",
        "fallback_model": "qwen-flash",
        "api_key": key
    }


def call_qwen_chat(
    messages: List[Dict[str, str]],
    api_key: str = "",
    provider_hint: str = "auto",
    temperature: float = 0.65,
    max_tokens: int = 2800
) -> Optional[str]:
    """
    Ejecuta una llamada al modelo Qwen del colaborador autenticado (con fallback de modelo si aplica).
    """
    cfg = resolve_qwen_endpoint(api_key, provider_hint)
    if not cfg:
        return None

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json"
    }
    models_to_try = [cfg["model"]]
    if cfg.get("fallback_model"):
        models_to_try.append(cfg["fallback_model"])

    for mdl in models_to_try:
        try:
            payload: Dict[str, Any] = {
                "model": mdl,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            resp = requests.post(cfg["url"], headers=headers, json=payload, timeout=28)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content", "").strip()
                    # Limpiar bloques <think>...</think> si el modelo razonador de Qwen los incluye
                    if "<think>" in content and "</think>" in content:
                        content = content.split("</think>", 1)[-1].strip()
                    if content:
                        return content
        except Exception:
            continue
    return None


def translate_text_free_fallback(text: str, target_lang: str = "es") -> str:
    """
    Traductor gratuito instantáneo sin API Key para garantizar que ningún video o artículo
    en inglés (u otro idioma) quede sin traducir al idioma elegido del podcast (es, en, pt).
    """
    clean = (text or "").strip()
    if not clean:
        return ""
    tl = target_lang if target_lang in ("es", "en", "pt") else "es"

    # Dividir en fragmentos seguros <= 1500 caracteres
    chunks: List[str] = []
    current = ""
    for sentence in clean.split(". "):
        if len(current) + len(sentence) < 1450:
            current = (current + ". " + sentence).strip(". ")
        else:
            if current:
                chunks.append(current + ".")
            current = sentence
    if current:
        chunks.append(current)

    translated_parts: List[str] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for ch in chunks[:6]:
        try:
            q = urllib.parse.quote(ch)
            url = f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl={tl}&q={q}"
            r = requests.get(url, headers=headers, timeout=7)
            if r.status_code == 200:
                arr = r.json()
                if isinstance(arr, list) and arr:
                    first = arr[0]
                    if isinstance(first, list) and first:
                        translated_parts.append(str(first[0]).strip())
                        continue
                    elif isinstance(first, str):
                        translated_parts.append(first.strip())
                        continue
        except Exception:
            pass
        try:
            q_short = urllib.parse.quote(ch[:480])
            r2 = requests.get(f"https://api.mymemory.translated.net/get?q={q_short}&langpair=en|{tl}", timeout=7)
            if r2.status_code == 200:
                t2 = (r2.json().get("responseData") or {}).get("translatedText", "")
                if t2:
                    translated_parts.append(t2.strip())
                    continue
        except Exception:
            pass
        translated_parts.append(ch)

    return " ".join(translated_parts).strip()


def translate_insights_dict(
    insights: Dict[str, Any],
    target_lang: str = "es",
    api_key: str = "",
    provider_hint: str = "auto"
) -> Dict[str, Any]:
    """
    Traduce y adapta culturalmente todos los hallazgos extraídos del video/artículo
    al idioma de salida del podcast (es, en, pt), usando Qwen si el usuario configuró su API Key,
    o el traductor gratuito integrado como respaldo garantizado.
    """
    tl = target_lang if target_lang in ("es", "en", "pt") else "es"
    lang_label = LANG_NAMES.get(tl, "Español")

    # 1. Si hay API Key de Qwen, hacer traducción contextual y curaduría editorial en una sola llamada JSON
    if api_key or os.environ.get("QWEN_API_KEY"):
        prompt = (
            f"Eres un productor ejecutivo de podcasts. Traduce y adapta TODO el siguiente análisis "
            f"al idioma {lang_label} ({tl}), con lenguaje natural, claro y humano (PROHIBIDO dejar frases en otro idioma). "
            f"Devuelve EXCLUSIVAMENTE un objeto JSON válido con las mismas claves: "
            f"title (string), summary (string), beyond_tech_points (array de 4 strings), "
            f"technical_deep_dive (array de 3 strings), practical_cases_and_ideas (array de 3 strings), "
            f"key_quotes (array de 2 strings).\n\n"
            f"CONTENIDO FUENTE:\n" + json.dumps({
                "title": insights.get("title", ""),
                "summary": insights.get("summary", ""),
                "beyond_tech_points": insights.get("beyond_tech_points", [])[:4],
                "technical_deep_dive": insights.get("technical_deep_dive", [])[:3],
                "practical_cases_and_ideas": insights.get("practical_cases_and_ideas", [])[:3],
                "key_quotes": insights.get("key_quotes", [])[:2],
            }, ensure_ascii=False)
        )
        raw_resp = call_qwen_chat(
            [{"role": "user", "content": prompt}],
            api_key=api_key,
            provider_hint=provider_hint,
            temperature=0.4,
            max_tokens=1800
        )
        if raw_resp:
            try:
                clean_json = raw_resp.strip()
                if "```" in clean_json:
                    clean_json = clean_json.split("```")[1]
                    if clean_json.startswith("json"):
                        clean_json = clean_json[4:]
                parsed = json.loads(clean_json.strip())
                out = dict(insights)
                for k in ("title", "summary", "beyond_tech_points", "technical_deep_dive", "practical_cases_and_ideas", "key_quotes"):
                    if parsed.get(k):
                        out[k] = parsed[k]
                out["playful_hooks"] = [
                    translate_text_free_fallback(h, tl) for h in insights.get("playful_hooks", [])
                ]
                return out
            except Exception:
                pass

    # 2. Respaldo gratuito garantizado sin API Key (traduce todos los campos al idioma meta)
    out = dict(insights)
    out["title"] = translate_text_free_fallback(insights.get("title", ""), tl)
    out["summary"] = translate_text_free_fallback(insights.get("summary", ""), tl)
    out["beyond_tech_points"] = [
        translate_text_free_fallback(p, tl) for p in insights.get("beyond_tech_points", [])
    ]
    out["technical_deep_dive"] = [
        translate_text_free_fallback(p, tl) for p in insights.get("technical_deep_dive", [])
    ]
    out["practical_cases_and_ideas"] = [
        translate_text_free_fallback(p, tl) for p in insights.get("practical_cases_and_ideas", [])
    ]
    out["key_quotes"] = [
        translate_text_free_fallback(p, tl) for p in insights.get("key_quotes", [])
    ]
    out["playful_hooks"] = [
        translate_text_free_fallback(p, tl) for p in insights.get("playful_hooks", [])
    ]
    return out
