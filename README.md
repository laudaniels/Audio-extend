# Audio-extend

Audio "extenden" (continuation) met [MusicGen Melody](https://huggingface.co/facebook/musicgen-melody)
(Meta's [audiocraft](https://github.com/facebookresearch/audiocraft)): een kort audiofragment
automatisch laten voortzetten in dezelfde stijl.

Bevat een CLI-testscript en een lokale web-GUI (Gradio).

## Belangrijk om te weten

- **Geen echte inpainting/repaint.** MusicGen genereert autoregressief, dus het kan een
  fragment alleen *verlengen* (na het einde), niet een gat middenin opvullen met context
  van beide kanten.
- **Modelcontext = 30 seconden.** Dit is een hard gegeven van het model, niet instelbaar.
  Input-fragmenten langer dan 30s worden door de GUI geweigerd. Langere output wordt
  intern in stukken (chunks) gegenereerd, telkens met het laatste stukje audio als context.
- **Standaard volledig offline.** Er wordt nooit naar huggingface.co gebeld, behalve
  expliciet via `--update` (CLI) of de "Controleer op model-update"-knop (GUI).
- **CPU-only is traag.** Zonder GPU kost het laden van musicgen-melody al ~2 minuten,
  en genereren van audio een veelvoud van de real-time duur.

## Installatie

Deze map is zelf de virtualenv-root (`pyvenv.cfg` staat hier). Om 'm elders opnieuw op
te zetten:

```bash
python3 -m venv .
source bin/activate
pip install -r requirements.txt
```

## Gebruik

### CLI — snel testen van continuation

```bash
source bin/activate
python scripts/test_continuation.py \
  --input samples/jouw_clip.wav --duration 15
```

Zonder `--input` genereert het script eerst zelf een korte seed-clip en extendeert die.
Zie `python scripts/test_continuation.py --help` voor alle opties (model, duur, prompt, etc.).

### GUI — drag & drop, progressbar, preview

```bash
source bin/activate
python scripts/gui.py
```

Open daarna `http://127.0.0.1:7860` in je browser. Functionaliteit:

- Drop-area voor een audiofragment (max 30s, langere fragmenten worden geweigerd).
- "Extend tijd" veld (standaard 30s).
- Live progressbar tijdens genereren.
- Preview-player + downloadknop voor het resultaat.
- "Verwerk origineel" — extend altijd vanaf het geüploade fragment.
- "Verwerk nieuw gegenereerd fragment" — ketting verder op het laatst gegenereerde resultaat.
- "Controleer op model-update" — eenmalige online check/download, daarna weer offline.

## Projectstructuur

```
scripts/
  test_continuation.py   CLI-testscript voor continuation
  gui.py                 Gradio-GUI
samples/                 (lokaal) input/seed-audio, niet in git
output/                  (lokaal) gegenereerde audio, niet in git
```
