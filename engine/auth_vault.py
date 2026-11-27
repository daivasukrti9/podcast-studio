import os
import json
import time
import secrets
import base64
from typing import Dict, Any, Optional

VAULT_FILENAME = ".local_users_vault.json"


def _get_vault_path(config_dir: str) -> str:
    return os.path.join(config_dir, VAULT_FILENAME)


def _load_vault(config_dir: str) -> Dict[str, Any]:
    path = _get_vault_path(config_dir)
    if not os.path.exists(path):
        return {"users": {}, "sessions": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("users", {})
            data.setdefault("sessions", {})
            return data
    except Exception:
        return {"users": {}, "sessions": {}}


def _save_vault(config_dir: str, data: Dict[str, Any]) -> None:
    os.makedirs(config_dir, exist_ok=True)
    path = _get_vault_path(config_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def decode_google_jwt_payload(credential_jwt: str) -> Optional[Dict[str, Any]]:
    """
    Decodifica el payload de un token JWT de Google Identity Services (Sign in with Google)
    para extraer email, nombre y foto del colaborador corporativo.
    """
    try:
        parts = credential_jwt.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        padding = "=" * ((4 - len(payload_b64) % 4) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return None


def login_or_register_gmail_user(
    config_dir: str,
    email: str,
    name: str = "",
    picture: str = ""
) -> Dict[str, Any]:
    """
    Inicia sesión o registra a un colaborador con su cuenta Gmail / Google Workspace,
    creando su espacio aislado para guardar su propia API Key de Qwen / Qwen3-TTS.
    """
    clean_email = (email or "").strip().lower()
    if not clean_email or "@" not in clean_email:
        raise ValueError("Ingresa un correo electrónico válido (Gmail o cuenta corporativa).")

    if not name.strip():
        local_part = clean_email.split("@")[0].replace(".", " ").replace("_", " ")
        name = local_part.title()

    vault = _load_vault(config_dir)
    user_record = vault["users"].get(clean_email, {
        "email": clean_email,
        "name": name.strip(),
        "picture": picture,
        "provider": "auto",
        "qwen_api_key": "",
        "use_qwen_tts": True,
        "created_at": int(time.time()),
        "episodes_created": 0
    })

    if name.strip():
        user_record["name"] = name.strip()
    if picture:
        user_record["picture"] = picture

    vault["users"][clean_email] = user_record

    token = secrets.token_urlsafe(24)
    vault["sessions"][token] = {
        "email": clean_email,
        "issued_at": int(time.time())
    }
    _save_vault(config_dir, vault)

    return {
        "session_token": token,
        "user": _public_user_view(user_record)
    }


def get_user_by_session(config_dir: str, session_token: str) -> Optional[Dict[str, Any]]:
    if not session_token:
        return None
    vault = _load_vault(config_dir)
    sess = vault["sessions"].get(session_token)
    if not sess:
        return None
    email = sess.get("email", "")
    return vault["users"].get(email)


def update_user_qwen_settings(
    config_dir: str,
    session_token: str,
    qwen_api_key: str,
    provider: str = "auto",
    use_qwen_tts: bool = True
) -> Optional[Dict[str, Any]]:
    vault = _load_vault(config_dir)
    sess = vault["sessions"].get(session_token)
    if not sess:
        return None
    email = sess["email"]
    user = vault["users"].get(email)
    if not user:
        return None

    if qwen_api_key.strip() != "********":
        user["qwen_api_key"] = qwen_api_key.strip()
    user["provider"] = provider or "auto"
    user["use_qwen_tts"] = bool(use_qwen_tts)
    vault["users"][email] = user
    _save_vault(config_dir, vault)
    return _public_user_view(user)


def logout_session(config_dir: str, session_token: str) -> None:
    if not session_token:
        return
    vault = _load_vault(config_dir)
    if session_token in vault["sessions"]:
        del vault["sessions"][session_token]
        _save_vault(config_dir, vault)


def _public_user_view(user: Dict[str, Any]) -> Dict[str, Any]:
    raw_key = user.get("qwen_api_key", "")
    masked = ""
    if raw_key:
        masked = raw_key[:6] + "..." + raw_key[-4:] if len(raw_key) > 10 else "Configurada"
    return {
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "picture": user.get("picture", ""),
        "provider": user.get("provider", "auto"),
        "has_api_key": bool(raw_key),
        "masked_api_key": masked,
        "use_qwen_tts": bool(user.get("use_qwen_tts", True))
    }
