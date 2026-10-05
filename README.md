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

### Met een NVIDIA GPU (aanbevolen voor bruikbare snelheid)

Dit is in eerste instantie op een CPU-only (WSL2) systeem gebouwd en getest. Met een
NVIDIA GPU draait alles aanzienlijk sneller — device-detectie (`torch.cuda.is_available()`)
gebeurt al automatisch in beide scripts, dus er hoeft niets in de code te veranderen.

1. **NVIDIA-driver + CUDA toolkit** staan al op het systeem (check met `nvidia-smi`).
2. **Venv aanmaken en torch met CUDA-support installeren — vóór de rest:**

   ```bash
   python3 -m venv .
   source bin/activate
   # Kies de juiste --index-url voor je CUDA-versie op https://pytorch.org/get-started/locally/
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   Controleer daarna dat CUDA echt gezien wordt, vóórdat je verder gaat:

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

3. **Pas daarna de rest installeren**, zodat pip de al-geïnstalleerde (CUDA-)torch niet
   overschrijft met een CPU-only versie:

   ```bash
   pip install -r requirements.txt --no-deps
   pip check   # meldt eventuele ontbrekende sub-dependencies, los die gericht op
   ```

   (`requirements.txt` is een `pip freeze` van de CPU-only ontwikkelomgeving; met
   `--no-deps` voorkom je dat de daarin gepinde CPU-torch je CUDA-install terugdraait.)

4. **xformers**: installeer een versie die bij je CUDA/torch-combinatie past
   (`pip install xformers`) zodat de C++/CUDA-extensies laden in plaats van de
   pure-Python fallback die op CPU werd gebruikt.
5. **Optioneel, voor meer throughput op GPU**: verhoog `CHUNK_SECONDS` in
   `scripts/gui.py` (stond bewust laag op 10s voor nette progress-feedback op een
   trage CPU) — minder chunks betekent minder overhead per `generate_continuation`-call.

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
