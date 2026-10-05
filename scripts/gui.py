#!/usr/bin/env python
"""
Lokale web-GUI voor MusicGen Melody continuation/extend.

Starten:
    python scripts/gui.py

Opent een server op http://127.0.0.1:7860 (alleen lokaal bereikbaar).

Werkt standaard volledig offline (HF_HUB_OFFLINE=1): er wordt nooit naar
huggingface.co gebeld, behalve als je zelf op "Controleer op model-update" klikt.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
import soundfile as sf
import torch

from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/musicgen-melody"
MAX_INPUT_SECONDS = 30.0
MODEL_MAX_DURATION = 30.0
CHUNK_SECONDS = 10.0
DEFAULT_EXTEND_SECONDS = 30.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[gui] device={DEVICE} model={MODEL_NAME} (offline cache)")
_t0 = time.time()
MODEL = MusicGen.get_pretrained(MODEL_NAME, device=DEVICE)
print(f"[gui] model geladen in {time.time() - _t0:.1f}s")


def _duration_seconds(path: str) -> float:
    return sf.info(path).duration


def _load_audio(path: str):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(wav.T), sr  # [C, T]


def _extend(source_path: str, extend_seconds: float, progress: gr.Progress) -> str:
    """Verlengt source_path met extend_seconds, in stappen van CHUNK_SECONDS.
    Elke stap blijft binnen het contextvenster van het model (<= MODEL_MAX_DURATION),
    zodat we zelf de voortgang per stap kunnen rapporteren."""
    wav, sr = _load_audio(source_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != MODEL.sample_rate:
        import julius
        wav = julius.resample_frac(wav, sr, MODEL.sample_rate)
        sr = MODEL.sample_rate

    working = wav
    total = max(extend_seconds, 0.0)
    done = 0.0
    progress(0.0, desc="Model aan het werk zetten...")

    def make_token_callback(step_seconds):
        def _cb(generated_tokens, tokens_to_generate):
            frac_within_chunk = generated_tokens / max(tokens_to_generate, 1)
            overall_done = done + frac_within_chunk * step_seconds
            progress(min(overall_done / total, 1.0) if total else 1.0,
                      desc=f"Genereren... {overall_done:.0f}/{total:.0f}s")
        return _cb

    while done < total - 1e-3:
        step = min(CHUNK_SECONDS, total - done)
        context_len = min(working.shape[-1] / sr, MODEL_MAX_DURATION - step - 0.1)
        context_len = max(context_len, 1.0)
        context_samples = int(context_len * sr)
        context = working[..., -context_samples:]

        MODEL.set_generation_params(duration=context_len + step)
        MODEL.set_custom_progress_callback(make_token_callback(step))
        out = MODEL.generate_continuation(
            context.unsqueeze(0).to(DEVICE),
            prompt_sample_rate=sr,
            descriptions=[None],
            progress=True,
        )[0].cpu()

        new_tail = out[..., context_samples:]
        working = torch.cat([working, new_tail], dim=-1)

        done += step
        progress(min(done / total, 1.0) if total else 1.0, desc=f"Genereren... {done:.0f}/{total:.0f}s")

    out_path = audio_write(
        str(OUTPUT_DIR / f"extend_{int(time.time())}"),
        working, sr, strategy="loudness", loudness_compressor=True,
    )
    return str(out_path)


def on_upload(path):
    """Validatie van de drop-area: wijst fragmenten > 30s af."""
    if path is None:
        return (gr.update(), None, None, gr.update(interactive=False), gr.update(interactive=False),
                gr.update(visible=False), "Geen bestand geselecteerd.")
    dur = _duration_seconds(path)
    if dur > MAX_INPUT_SECONDS + 0.05:
        return (gr.update(value=None), None, None, gr.update(interactive=False), gr.update(interactive=False),
                gr.update(visible=False),
                f"⚠️ Fragment is {dur:.1f}s — maximaal {MAX_INPUT_SECONDS:.0f}s toegestaan. Kies een korter fragment.")
    return (gr.update(), path, None, gr.update(interactive=True), gr.update(interactive=False),
            gr.update(visible=False), f"✅ {dur:.1f}s geladen, klaar om te verwerken.")


def process(source_path, extend_seconds, progress=gr.Progress()):
    if not source_path:
        return gr.update(), gr.update(), gr.update(visible=False), gr.update(interactive=False), \
            "Geen geldige audio-bron beschikbaar."
    try:
        extend_seconds = float(extend_seconds)
    except (TypeError, ValueError):
        extend_seconds = DEFAULT_EXTEND_SECONDS
    out_path = _extend(source_path, extend_seconds, progress)
    return out_path, out_path, gr.update(value=out_path, visible=True), gr.update(interactive=True), \
        f"✅ Klaar — {Path(out_path).name}"


def check_update(progress=gr.Progress()):
    progress(0.0, desc="Verbinden met Hugging Face Hub...")
    env = os.environ.copy()
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    code = (
        "from audiocraft.models import MusicGen;"
        f"MusicGen.get_pretrained({MODEL_NAME!r}, device='cpu')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return "⚠️ Time-out bij het controleren op updates (geen of een trage internetverbinding?)."
    if result.returncode != 0:
        return f"⚠️ Check mislukt:\n{result.stderr.strip()[-800:]}"

    progress(0.8, desc="Model herladen...")
    global MODEL
    MODEL = MusicGen.get_pretrained(MODEL_NAME, device=DEVICE)
    return "✅ Model is up-to-date (eventuele nieuwe versie is gedownload) en opnieuw geladen."


with gr.Blocks(title="MusicGen Melody — Extend") as demo:
    gr.Markdown(
        f"## MusicGen Melody — Extend\n"
        f"Model: `{MODEL_NAME}` · Device: `{DEVICE}` · Model-contextvenster: {MODEL_MAX_DURATION:.0f}s"
    )

    with gr.Row():
        update_btn = gr.Button("🔄 Controleer op model-update")
    update_status = gr.Markdown("")

    audio_in = gr.Audio(
        sources=["upload"], type="filepath",
        label=f"Sleep audiofragment hierheen (max {MAX_INPUT_SECONDS:.0f}s)",
    )
    upload_status = gr.Markdown("")

    extend_seconds = gr.Number(
        value=DEFAULT_EXTEND_SECONDS, minimum=1, maximum=60, step=1,
        label="Extend tijd (seconden)",
    )

    with gr.Row():
        btn_original = gr.Button("🔁 Verwerk origineel", variant="primary", interactive=False)
        btn_continue = gr.Button("➕ Verwerk nieuw gegenereerd fragment", interactive=False)

    status = gr.Markdown("")
    preview = gr.Audio(type="filepath", label="Preview", interactive=False)
    download_btn = gr.DownloadButton("⬇️ Download resultaat", visible=False)

    original_state = gr.State(None)
    current_state = gr.State(None)

    audio_in.change(
        fn=on_upload, inputs=audio_in,
        outputs=[audio_in, original_state, current_state, btn_original, btn_continue, download_btn, upload_status],
    )

    update_btn.click(fn=check_update, outputs=update_status)

    btn_original.click(
        fn=process, inputs=[original_state, extend_seconds],
        outputs=[current_state, preview, download_btn, btn_continue, status],
    )
    btn_continue.click(
        fn=process, inputs=[current_state, extend_seconds],
        outputs=[current_state, preview, download_btn, btn_continue, status],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
