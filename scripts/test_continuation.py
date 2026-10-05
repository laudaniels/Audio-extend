#!/usr/bin/env python
"""
Testscript voor MusicGen continuation (audio "extenden").

Gebruik:
  python scripts/test_continuation.py
      -> genereert zelf eerst een korte seed-clip en extendeert die.

  python scripts/test_continuation.py --input samples/mijn_clip.wav --duration 15
      -> extendeert een bestaand audiofragment tot ~15 seconden.

Let op: geen GPU gedetecteerd in deze omgeving -> dit draait op CPU en is traag.
Voor een snelle smoke test: --model facebook/musicgen-small.

Netwerk: standaard draait dit script volledig offline (HF_HUB_OFFLINE=1), dus er
wordt nooit naar huggingface.co gebeld als het model al lokaal in de cache zit.
Gebruik --update om die check/download van een nieuwere modelversie wel toe te staan.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Moet vóór de audiocraft/huggingface_hub import gezet worden, anders heeft het geen effect.
if "--update" not in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"

import torch

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write, audio_read

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None,
                         help="Pad naar input audiofragment (wav/mp3). Als leeg: er wordt eerst een seed-clip gegenereerd.")
    parser.add_argument("--model", type=str, default="facebook/musicgen-melody",
                         help="Model checkpoint (default: facebook/musicgen-melody; gebruik musicgen-small voor een snelle test).")
    parser.add_argument("--seed-prompt", type=str, default="upbeat lo-fi hip hop with piano and soft drums",
                         help="Tekstprompt om de seed-clip mee te genereren (alleen gebruikt als --input ontbreekt).")
    parser.add_argument("--seed-duration", type=float, default=4.0,
                         help="Lengte van de zelf-gegenereerde seed-clip in seconden.")
    parser.add_argument("--continuation-prompt", type=str, default=None,
                         help="Tekstprompt die de voortzetting stuurt (optioneel, mag leeg blijven).")
    parser.add_argument("--duration", type=float, default=12.0,
                         help="Totale lengte van het eindresultaat in seconden (inclusief het inputstuk).")
    parser.add_argument("--extend-stride", type=float, default=18.0,
                         help="Stapgrootte voor het intern verlengen voorbij het model-contextvenster (<= model max duration).")
    parser.add_argument("--outdir", type=str, default=str(ROOT / "output"))
    parser.add_argument("--update", action="store_true",
                         help="Sta een check/download bij HF Hub toe (anders draait alles offline op de lokale cache).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")
    print(f"[info] model  = {args.model}")
    print(f"[info] netwerk = {'online (--update)' if args.update else 'offline (HF_HUB_OFFLINE=1)'}")

    t0 = time.time()
    model = MusicGen.get_pretrained(args.model, device=device)
    print(f"[info] model geladen in {time.time() - t0:.1f}s")

    sample_rate = model.sample_rate

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input bestand niet gevonden: {input_path}")
        prompt_wav, prompt_sr = audio_read(str(input_path))
        print(f"[info] input geladen: {input_path} ({prompt_wav.shape[-1] / prompt_sr:.2f}s @ {prompt_sr}Hz)")
    else:
        print(f"[info] geen --input opgegeven, genereer eerst een seed-clip ({args.seed_duration}s)...")
        model.set_generation_params(duration=args.seed_duration)
        t0 = time.time()
        seed_wav = model.generate([args.seed_prompt], progress=True)[0]
        print(f"[info] seed-clip gegenereerd in {time.time() - t0:.1f}s")
        seed_path = audio_write(
            str(ROOT / "samples" / "seed"), seed_wav.cpu(), sample_rate,
            strategy="loudness", loudness_compressor=True,
        )
        print(f"[info] seed-clip opgeslagen als {seed_path}")
        prompt_wav, prompt_sr = seed_wav, sample_rate

    if prompt_sr != sample_rate:
        import julius
        prompt_wav = julius.resample_frac(prompt_wav, prompt_sr, sample_rate)

    # Mono -> wat MusicGen verwacht, als het bestand stereo is nemen we het gemiddelde.
    if prompt_wav.dim() == 2 and prompt_wav.shape[0] > 1:
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
    if prompt_wav.dim() == 1:
        prompt_wav = prompt_wav.unsqueeze(0)

    model.set_generation_params(duration=args.duration, extend_stride=min(args.extend_stride, args.duration))

    descriptions = [args.continuation_prompt] if args.continuation_prompt else [None]

    print(f"[info] start continuation naar {args.duration}s totale lengte...")
    t0 = time.time()
    output = model.generate_continuation(
        prompt_wav.unsqueeze(0).to(device),
        prompt_sample_rate=sample_rate,
        descriptions=descriptions,
        progress=True,
    )
    elapsed = time.time() - t0
    print(f"[info] continuation klaar in {elapsed:.1f}s")

    out_path = audio_write(
        str(outdir / "continuation"), output[0].cpu(), sample_rate,
        strategy="loudness", loudness_compressor=True,
    )
    print(f"[done] resultaat opgeslagen als: {out_path}")


if __name__ == "__main__":
    main()
