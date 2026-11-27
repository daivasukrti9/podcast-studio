import os
import io
import re
import wave
import math
import base64
import random
import asyncio
import subprocess
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import signal
import requests

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"

try:
    import edge_tts
    HAS_EDGE_TTS = True
except Exception:
    HAS_EDGE_TTS = False

try:
    from kokoro import KPipeline  # type: ignore
    HAS_KOKORO_LOCAL = True
except Exception:
    HAS_KOKORO_LOCAL = False


SAMPLE_RATE = 24000


def _db_to_linear(db_val: float) -> float:
    return float(10.0 ** (db_val / 20.0))


def decode_audio_bytes_via_ffmpeg(raw_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    if not raw_bytes:
        return np.array([], dtype=np.float32)

    try:
        cmd = [
            FFMPEG_EXE,
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-ac", "1",
            "pipe:1"
        ]
        proc = subprocess.run(cmd, input=bytes(raw_bytes), capture_output=True, timeout=25)
        if proc.returncode == 0 and proc.stdout:
            return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            raw_pcm = wf.readframes(wf.getnframes())
            return np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return np.array([], dtype=np.float32)


def apply_vocal_color_shaping(
    samples: np.ndarray,
    tone: str = "calido",
    exaggeration: float = 0.45,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    if samples.size == 0:
        return samples

    x = samples.astype(np.float32)
    exag = float(np.clip(exaggeration, 0.15, 0.85))

    if tone in ("entusiasta", "complice"):
        sos_bp = signal.butter(2, [2200.0 / (sr * 0.5), 4600.0 / (sr * 0.5)], btype="bandpass", output="sos")
        presence = signal.sosfilt(sos_bp, x)
        x = x + presence * (0.14 * exag)
    elif tone == "reflexivo":
        sos_warm = signal.butter(2, [140.0 / (sr * 0.5), 380.0 / (sr * 0.5)], btype="bandpass", output="sos")
        warmth = signal.sosfilt(sos_warm, x)
        x = x + warmth * (0.18 * exag)
    elif tone == "tecnico":
        sos_mid = signal.butter(2, [1300.0 / (sr * 0.5), 3100.0 / (sr * 0.5)], btype="bandpass", output="sos")
        clarity = signal.sosfilt(sos_mid, x)
        x = x + clarity * (0.12 * exag)

    return x.astype(np.float32)


def apply_ai_free_dsp_voice_chain(
    samples: np.ndarray,
    sr: int = SAMPLE_RATE,
    tone: str = "calido",
    exaggeration: float = 0.45
) -> np.ndarray:
    if samples.size == 0:
        return samples

    x = samples.astype(np.float32)
    x = x - np.mean(x)

    sos_hp = signal.butter(2, 80.0 / (sr * 0.5), btype="highpass", output="sos")
    x = signal.sosfilt(sos_hp, x)

    x = apply_vocal_color_shaping(x, tone=tone, exaggeration=exaggeration, sr=sr)

    target_rms = _db_to_linear(-16.0)
    current_rms = float(np.sqrt(np.mean(x ** 2) + 1e-9))
    if current_rms > 1e-6:
        x = x * (target_rms / current_rms)

    x = np.tanh(x * 1.12) / np.tanh(1.12)
    return np.clip(x, -0.96, 0.96).astype(np.float32)


def generate_studio_room_tone(duration_sec: float, sr: int = SAMPLE_RATE, target_db: float = -42.0) -> np.ndarray:
    n_samples = max(1, int(duration_sec * sr))
    rng = np.random.default_rng(42)
    white = rng.standard_normal(n_samples, dtype=np.float32)
    sos = signal.butter(2, [90.0 / (sr * 0.5), 950.0 / (sr * 0.5)], btype="bandpass", output="sos")
    filtered = signal.sosfilt(sos, white)
    target_rms = _db_to_linear(target_db)
    curr_rms = float(np.sqrt(np.mean(filtered ** 2) + 1e-12))
    if curr_rms > 1e-9:
        filtered = filtered * (target_rms / curr_rms)
    return filtered.astype(np.float32)


def generate_warm_ambient_bed(
    duration_sec: float,
    style: str = "warm_ambient_chords",
    sr: int = SAMPLE_RATE,
    target_lufs: float = -30.0
) -> np.ndarray:
    n_samples = max(1, int(duration_sec * sr))
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    if style == "pastel_acoustic_warm":
        chord_a = [174.61, 220.00, 261.63, 329.63]
        chord_b = [130.81, 196.00, 246.94, 329.63]
    elif style == "cinematic_pulse_soft":
        chord_a = [146.83, 220.00, 293.66, 369.99]
        chord_b = [164.81, 246.94, 329.63, 415.30]
    else:
        chord_a = [146.83, 220.00, 261.63, 329.63]
        chord_b = [116.54, 174.61, 220.00, 293.66]

    bed = np.zeros(n_samples, dtype=np.float32)
    morph = 0.5 * (1.0 + np.sin(2.0 * np.pi * (1.0 / 16.0) * t))

    for f1, f2 in zip(chord_a, chord_b):
        w1 = np.sin(2.0 * np.pi * f1 * t)
        w2 = np.sin(2.0 * np.pi * f2 * t)
        bed += (1.0 - morph) * w1 + morph * w2

    sos = signal.butter(2, 420.0 / (sr * 0.5), btype="lowpass", output="sos")
    bed = signal.sosfilt(sos, bed)

    target_rms = _db_to_linear(target_lufs)
    current_rms = float(np.sqrt(np.mean(bed ** 2) + 1e-9))
    if current_rms > 1e-6:
        bed = bed * (target_rms / current_rms)

    room_tone = generate_studio_room_tone(duration_sec, sr=sr, target_db=-42.0)
    return (bed + room_tone).astype(np.float32)


def generate_micro_transition_chime(sr: int = SAMPLE_RATE) -> np.ndarray:
    dur = 1.4
    n = int(dur * sr)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
    notes = [(0.0, 261.63), (0.18, 329.63), (0.36, 392.00), (0.55, 523.25)]
    out = np.zeros(n, dtype=np.float32)

    for start_t, freq in notes:
        mask = t >= start_t
        local_t = t[mask] - start_t
        env = np.exp(-4.2 * local_t) * (1.0 - np.exp(-50.0 * local_t))
        tone = np.sin(2.0 * np.pi * freq * local_t)
        out[mask] += env * tone

    target_rms = _db_to_linear(-25.0)
    rms = float(np.sqrt(np.mean(out ** 2) + 1e-9))
    if rms > 1e-6:
        out = out * (target_rms / rms)
    return out.astype(np.float32)


def mix_with_auto_ducking(
    voice_track: np.ndarray,
    bed_track: np.ndarray,
    transition_positions_samples: List[int],
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    n = len(voice_track)
    if len(bed_track) < n:
        reps = int(math.ceil(n / len(bed_track)))
        bed_track = np.tile(bed_track, reps)[:n]
    else:
        bed_track = bed_track[:n].copy()

    chime = generate_micro_transition_chime(sr)
    for pos in transition_positions_samples:
        if 0 <= pos < n:
            end_pos = min(n, pos + len(chime))
            bed_track[pos:end_pos] += chime[: end_pos - pos]

    win = max(1, int(sr * 0.15))
    voice_energy = np.sqrt(np.convolve(voice_track ** 2, np.ones(win) / win, mode="same"))
    is_speaking = (voice_energy > 0.012).astype(np.float32)

    smooth_win = max(1, int(sr * 0.40))
    duck_curve = np.convolve(is_speaking, np.ones(smooth_win) / smooth_win, mode="same")
    bed_gain = 0.95 - (0.67 * np.clip(duck_curve, 0.0, 1.0))

    mixed = voice_track + (bed_track * bed_gain)
    return np.clip(mixed, -0.98, 0.98).astype(np.float32)


def _clean_tags_for_continuous_speech(text: str) -> str:
    """
    Limpia etiquetas para lectura continua en una sola toma orgánica sin cortes ni muletillas forzadas.
    """
    t = re.sub(r"\[risa\s*ligera\]", "", text, flags=re.IGNORECASE)
    t = re.sub(r"\[pausa_corta\]", ", ", t, flags=re.IGNORECASE)
    t = re.sub(r"\[(?:pausa|risa|énfasis|suspiro)[^\]]*\]", ", ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _synthesize_qwen3_tts(
    text: str,
    qwen_voice: str,
    api_key: str
) -> np.ndarray:
    """
    Conector directo para Qwen3-TTS (`qwen3-tts-flash` / `qwen-tts`) en Alibaba Model Studio Internacional
    usando la API Key personal del colaborador autenticado con Gmail.
    """
    if not api_key or api_key.startswith(("gsk_", "sk-or-")):
        return np.array([], dtype=np.float32)

    clean_txt = _clean_tags_for_continuous_speech(text)
    if not clean_txt:
        return np.array([], dtype=np.float32)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    endpoints = [
        ("https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation", "qwen3-tts-flash"),
        ("https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation", "qwen-tts"),
    ]
    for url, mdl in endpoints:
        try:
            payload = {
                "model": mdl,
                "input": {
                    "text": clean_txt,
                    "voice": qwen_voice or "Serena"
                }
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=18)
            if resp.status_code == 200:
                j = resp.json()
                audio_obj = (j.get("output") or {}).get("audio") or {}
                audio_url = audio_obj.get("url")
                audio_b64 = audio_obj.get("data")
                if audio_url:
                    ar = requests.get(audio_url, timeout=15)
                    if ar.status_code == 200 and ar.content:
                        pcm = decode_audio_bytes_via_ffmpeg(ar.content, SAMPLE_RATE)
                        if pcm.size > 0:
                            return pcm
                if audio_b64:
                    raw_b = base64.b64decode(audio_b64)
                    pcm = decode_audio_bytes_via_ffmpeg(raw_b, SAMPLE_RATE)
                    if pcm.size > 0:
                        return pcm
        except Exception:
            continue
    return np.array([], dtype=np.float32)


async def _synthesize_single_take_edge(
    text: str,
    voice_id: str,
    rate: str = "+0%",
    pitch: str = "+0Hz"
) -> np.ndarray:
    """
    Sintetiza la intervención completa en UNA SOLA TOMA CONTINUA (sin cortar frases a la mitad),
    conservando la respiración, prosodia natural y fluidez que hizo que el final sonara súper humano.
    """
    clean_txt = _clean_tags_for_continuous_speech(text)
    if not HAS_EDGE_TTS or not clean_txt:
        return np.array([], dtype=np.float32)

    for _ in range(2):
        try:
            communicate = edge_tts.Communicate(text=clean_txt, voice=voice_id, rate=rate, pitch=pitch)
            mp3_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes.extend(chunk["data"])
            if mp3_bytes:
                pcm = decode_audio_bytes_via_ffmpeg(bytes(mp3_bytes), SAMPLE_RATE)
                if pcm.size > 0:
                    return pcm
        except Exception:
            await asyncio.sleep(0.25)
    return np.array([], dtype=np.float32)


async def _synthesize_line_expressive(
    line: Dict[str, Any],
    qwen_api_key: str = "",
    use_qwen_tts: bool = True
) -> np.ndarray:
    raw_text = line.get("text", "").strip()
    if not raw_text:
        return np.array([], dtype=np.float32)

    # 1. Intentar Qwen3-TTS si el usuario conectó su API Key de Alibaba Model Studio / Qwen
    if qwen_api_key and use_qwen_tts and not qwen_api_key.startswith(("gsk_", "sk-or-")):
        qwen_pcm = await asyncio.to_thread(
            _synthesize_qwen3_tts,
            raw_text,
            line.get("qwen_voice", "Serena"),
            qwen_api_key
        )
        if qwen_pcm.size > 0:
            return qwen_pcm

    # 2. Toma continua de alta naturalidad con las 24 voces Multilingües HD (ES / EN / PT)
    return await _synthesize_single_take_edge(
        text=raw_text,
        voice_id=line.get("voice_id", "es-MX-DaliaNeural"),
        rate=line.get("rate", "+0%"),
        pitch=line.get("pitch", "+0Hz")
    )


def save_wav_file(filepath: str, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def generate_srt_subtitles(timed_segments: List[Dict[str, Any]]) -> str:
    def fmt_ts(sec: float) -> str:
        ms = int(round((sec - int(sec)) * 1000))
        s = int(sec) % 60
        m = (int(sec) // 60) % 60
        h = int(sec) // 3600
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for idx, seg in enumerate(timed_segments, start=1):
        blocks.append(
            f"{idx}\n"
            f"{fmt_ts(seg['audio_start_sec'])} --> {fmt_ts(seg['audio_end_sec'])}\n"
            f"[{seg['speaker_name']}] {seg['text']}\n"
        )
    return "\n".join(blocks)


def generate_castopod_rss_xml(episode_meta: Dict[str, Any], audio_url: str) -> str:
    now_rfc = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    title = episode_meta.get("episode_title", "Episodio Podcast")
    show_name = episode_meta.get("show_name", "Podcast Studio")
    lang_code = episode_meta.get("language", "es")
    desc = episode_meta.get("core_intention", "Podcast corporativo y comunitario")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>{show_name}</title>
    <description>{desc}</description>
    <language>{lang_code}</language>
    <lastBuildDate>{now_rfc}</lastBuildDate>
    <item>
      <title>{title}</title>
      <description>{desc}</description>
      <pubDate>{now_rfc}</pubDate>
      <enclosure url="{audio_url}" type="audio/wav" />
      <itunes:explicit>false</itunes:explicit>
    </item>
  </channel>
</rss>
"""


async def render_episode_master_audio(
    audited_script: List[Dict[str, Any]],
    output_dir: str,
    episode_slug: str,
    bed_style: str = "warm_ambient_chords",
    use_tts: bool = True,
    qwen_api_key: str = "",
    use_qwen_tts: bool = True
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    sem = asyncio.Semaphore(5)

    async def synth_one(line_obj: Dict[str, Any]) -> np.ndarray:
        if not use_tts:
            return np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)
        async with sem:
            return await _synthesize_line_expressive(
                line_obj, qwen_api_key=qwen_api_key, use_qwen_tts=use_qwen_tts
            )

    synthesized_voices = await asyncio.gather(*(synth_one(l) for l in audited_script))

    voice_chunks: List[np.ndarray] = []
    transition_positions: List[int] = []
    timed_segments: List[Dict[str, Any]] = []

    current_sample = 0

    for idx, (line, raw_voice) in enumerate(zip(audited_script, synthesized_voices)):
        tone = line.get("tone", "calido")
        exag = float(line.get("exaggeration", 0.45))
        raw_txt = line.get("text", "")

        if line.get("sfx_transition_approved"):
            transition_positions.append(current_sample)
            intro_pad = np.zeros(int(0.75 * SAMPLE_RATE), dtype=np.float32)
            voice_chunks.append(intro_pad)
            current_sample += len(intro_pad)

        if raw_voice.size == 0:
            raw_voice = np.zeros(int(SAMPLE_RATE * 1.2), dtype=np.float32)

        clean_voice = apply_ai_free_dsp_voice_chain(
            raw_voice, SAMPLE_RATE, tone=tone, exaggeration=exag
        )

        # Solapamiento suave únicamente en confirmaciones cortas (is_backchannel) y nunca en el primer bloque
        is_bc = bool(line.get("is_backchannel", False))
        overlap_ms = int(line.get("overlap_ms", 130)) if (is_bc and idx >= 4) else 0

        if overlap_ms > 0 and len(voice_chunks) >= 2 and not line.get("sfx_transition_approved"):
            prev_pause = voice_chunks.pop()
            current_sample -= len(prev_pause)
            prev_voice = voice_chunks[-1]
            overlap_samples = min(int((overlap_ms / 1000.0) * SAMPLE_RATE), len(prev_voice) // 3, len(clean_voice) // 3)
            if overlap_samples > 0:
                prev_voice[-overlap_samples:] = np.clip(
                    prev_voice[-overlap_samples:] + clean_voice[:overlap_samples] * 0.85,
                    -0.96, 0.96
                )
                clean_voice = clean_voice[overlap_samples:]

        start_sec = round(current_sample / SAMPLE_RATE, 2)
        voice_chunks.append(clean_voice)
        current_sample += len(clean_voice)
        end_sec = round(current_sample / SAMPLE_RATE, 2)

        seg_info = dict(line)
        seg_info["audio_start_sec"] = start_sec
        seg_info["audio_end_sec"] = end_sec
        timed_segments.append(seg_info)

        if "[pausa_corta]" in raw_txt.lower() or "..." in raw_txt or tone == "reflexivo":
            silence_ms = random.randint(520, 740)
        elif is_bc:
            silence_ms = random.randint(140, 220)
        else:
            silence_ms = random.randint(210, 360)

        pause_chunk = np.zeros(int((silence_ms / 1000.0) * SAMPLE_RATE), dtype=np.float32)
        voice_chunks.append(pause_chunk)
        current_sample += len(pause_chunk)

    full_voice = np.concatenate(voice_chunks) if voice_chunks else np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    total_dur_sec = round(len(full_voice) / SAMPLE_RATE, 2)

    bed_track = generate_warm_ambient_bed(
        duration_sec=total_dur_sec + 1.2,
        style=bed_style,
        sr=SAMPLE_RATE,
        target_lufs=-30.0
    )
    tail_pad = np.zeros(int(1.2 * SAMPLE_RATE), dtype=np.float32)
    full_voice_padded = np.concatenate([full_voice, tail_pad])

    master_mix = mix_with_auto_ducking(
        voice_track=full_voice_padded,
        bed_track=bed_track,
        transition_positions_samples=transition_positions,
        sr=SAMPLE_RATE
    )

    wav_filename = f"{episode_slug}.wav"
    srt_filename = f"{episode_slug}.srt"
    rss_filename = f"{episode_slug}_rss.xml"

    wav_path = os.path.join(output_dir, wav_filename)
    srt_path = os.path.join(output_dir, srt_filename)

    save_wav_file(wav_path, master_mix, SAMPLE_RATE)

    srt_content = generate_srt_subtitles(timed_segments)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    n_bars = 62
    chunk_len = max(1, len(master_mix) // n_bars)
    waveform_bars: List[int] = []
    for b in range(n_bars):
        sl = master_mix[b * chunk_len : (b + 1) * chunk_len]
        rms_b = float(np.sqrt(np.mean(sl ** 2) + 1e-9)) if sl.size > 0 else 0.02
        height_px = int(np.clip(6 + rms_b * 145, 5, 34))
        waveform_bars.append(height_px)

    return {
        "wav_filename": wav_filename,
        "srt_filename": srt_filename,
        "rss_filename": rss_filename,
        "duration_seconds": round(len(master_mix) / SAMPLE_RATE, 1),
        "duration_minutes": round(len(master_mix) / (SAMPLE_RATE * 60.0), 1),
        "waveform_bars_62": waveform_bars,
        "timed_segments": timed_segments
    }


def process_uploaded_voice_without_ai(
    raw_bytes: bytes,
    output_dir: str,
    slug: str,
    add_ambient_bed: bool = True,
    bed_style: str = "warm_ambient_chords"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    samples = decode_audio_bytes_via_ffmpeg(raw_bytes, SAMPLE_RATE)
    if samples.size == 0:
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

    clean_voice = apply_ai_free_dsp_voice_chain(samples, SAMPLE_RATE)

    if add_ambient_bed:
        dur = len(clean_voice) / SAMPLE_RATE
        bed = generate_warm_ambient_bed(dur + 1.0, style=bed_style, sr=SAMPLE_RATE, target_lufs=-30.0)
        final_audio = mix_with_auto_ducking(clean_voice, bed, [0], sr=SAMPLE_RATE)
    else:
        final_audio = clean_voice

    out_name = f"{slug}_dsp_clean.wav"
    out_path = os.path.join(output_dir, out_name)
    save_wav_file(out_path, final_audio, SAMPLE_RATE)

    return {
        "wav_filename": out_name,
        "duration_seconds": round(len(final_audio) / SAMPLE_RATE, 2),
        "voice_lufs": -16.0,
        "bed_lufs": -24.0 if add_ambient_bed else None
    }


def _split_voice_turns(samples: np.ndarray, sr: int = SAMPLE_RATE, target_turns: int = 3) -> List[np.ndarray]:
    if samples.size == 0:
        return []

    win = max(1, int(sr * 0.12))
    energy = np.sqrt(np.convolve(samples ** 2, np.ones(win) / win, mode="same"))
    thresh = max(float(np.percentile(energy, 25)) * 1.3, 0.012)
    is_silent = energy < thresh

    min_pause_samples = int(sr * 0.45)
    min_turn_samples = int(sr * 1.2)

    turns: List[np.ndarray] = []
    start_idx = 0
    silent_run = 0

    for i, sil in enumerate(is_silent):
        if sil:
            silent_run += 1
        else:
            if silent_run >= min_pause_samples and (i - start_idx) >= min_turn_samples:
                chunk = samples[start_idx : i - (silent_run // 2)]
                if chunk.size > int(sr * 0.5):
                    turns.append(chunk)
                start_idx = i
            silent_run = 0

    rem = samples[start_idx:]
    if rem.size > int(sr * 0.4):
        turns.append(rem)

    if len(turns) <= 1 and target_turns > 1 and samples.size > sr * 3:
        splits = np.array_split(samples, min(target_turns, 4))
        turns = [s for s in splits if s.size > 0]

    return turns


async def process_human_and_ai_dialogue(
    raw_bytes: bytes,
    script_lines: List[Dict[str, Any]],
    ai_persona: Dict[str, Any],
    output_dir: str,
    slug: str,
    bed_style: str = "pastel_acoustic_warm",
    qwen_api_key: str = "",
    use_qwen_tts: bool = True
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    samples = decode_audio_bytes_via_ffmpeg(raw_bytes, SAMPLE_RATE)
    if samples.size == 0:
        samples = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

    clean_user_voice = apply_ai_free_dsp_voice_chain(samples, SAMPLE_RATE)

    ai_lines = [
        l for l in (script_lines or [])
        if l.get("speaker_name", "").lower() == ai_persona.get("name", "").lower() and not l.get("is_backchannel")
    ]
    if not ai_lines and script_lines:
        ai_lines = [l for l in script_lines[1::2] if not l.get("is_backchannel")]
    if not ai_lines:
        ai_lines = [
            {"text": f"Qué gran punto acabas de mencionar. Desde mi mirada como {ai_persona.get('role', 'co-anfitrión')}, eso conecta directo con nuestra comunidad.", "tone": "calido"},
            {"text": "Totalmente de acuerdo. Sigamos profundizando en cómo llevar esta idea a la práctica en el equipo.", "tone": "calido"}
        ]

    user_turns = _split_voice_turns(clean_user_voice, SAMPLE_RATE, target_turns=max(2, len(ai_lines)))

    dialogue_chunks: List[np.ndarray] = []
    max_pairs = max(len(user_turns), min(len(ai_lines), 6))

    for i in range(max_pairs):
        if i < len(user_turns):
            dialogue_chunks.append(apply_ai_free_dsp_voice_chain(user_turns[i], SAMPLE_RATE))
            pause_ms = random.randint(200, 340)
            dialogue_chunks.append(np.zeros(int((pause_ms / 1000.0) * SAMPLE_RATE), dtype=np.float32))

        if i < len(ai_lines):
            line_obj = dict(ai_lines[i])
            line_obj.setdefault("voice_id", ai_persona.get("voice_id", "es-MX-JorgeNeural"))
            line_obj.setdefault("qwen_voice", ai_persona.get("qwen_voice", "Ethan"))
            line_obj.setdefault("rate", ai_persona.get("rate", "+1%"))
            line_obj.setdefault("pitch", ai_persona.get("pitch", "+0Hz"))
            ai_raw = await _synthesize_line_expressive(line_obj, qwen_api_key=qwen_api_key, use_qwen_tts=use_qwen_tts)
            if ai_raw.size > 0:
                dialogue_chunks.append(
                    apply_ai_free_dsp_voice_chain(
                        ai_raw, SAMPLE_RATE,
                        tone=line_obj.get("tone", "calido"),
                        exaggeration=float(line_obj.get("exaggeration", 0.45))
                    )
                )
                pause_ms = random.randint(210, 350)
                dialogue_chunks.append(np.zeros(int((pause_ms / 1000.0) * SAMPLE_RATE), dtype=np.float32))

    combined_voice = np.concatenate(dialogue_chunks) if dialogue_chunks else clean_user_voice
    dur = len(combined_voice) / SAMPLE_RATE
    bed = generate_warm_ambient_bed(dur + 1.2, style=bed_style, sr=SAMPLE_RATE, target_lufs=-30.0)
    final_mix = mix_with_auto_ducking(combined_voice, bed, [0], sr=SAMPLE_RATE)

    out_name = f"{slug}_dialogo_ia.wav"
    out_path = os.path.join(output_dir, out_name)
    save_wav_file(out_path, final_mix, SAMPLE_RATE)

    n_bars = 62
    chunk_len = max(1, len(final_mix) // n_bars)
    waveform_bars: List[int] = []
    for b in range(n_bars):
        sl = final_mix[b * chunk_len : (b + 1) * chunk_len]
        rms_b = float(np.sqrt(np.mean(sl ** 2) + 1e-9)) if sl.size > 0 else 0.02
        waveform_bars.append(int(np.clip(6 + rms_b * 145, 5, 34)))

    return {
        "wav_filename": out_name,
        "duration_seconds": round(len(final_mix) / SAMPLE_RATE, 2),
        "user_turns_detected": len(user_turns),
        "ai_turns_interleaved": min(len(ai_lines), max_pairs),
        "ai_partner_name": ai_persona.get("name", "Agente Preconfigurado"),
        "waveform_bars_62": waveform_bars
    }
