"""Transcribe every WAV with the Vosk English model into a {stem: transcript} pickle."""

import os
import json
import wave
import pickle

from tqdm import tqdm
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)  # silence Vosk's verbose kaldi logging

ROOT = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data")
AUDIO_DIR = os.path.join(ROOT, "AudioFiles")
MODEL_DIR = os.path.join(ROOT, "vosk-model-en-us-0.22")
OUT_PATH = os.path.join(ROOT, "all__video_vosk_audioMap.p")
SAVE_EVERY = 25  # checkpoint frequency


def transcribe(rec, wav_path):
    """Stream a 16 kHz mono PCM wav through the recognizer; return the joined transcript."""
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        wf.close()
        raise ValueError(f"expected 16-bit mono wav, got ch={wf.getnchannels()} width={wf.getsampwidth()}")
    rec.SetWords(False)
    parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    wf.close()
    return " ".join(p for p in parts if p).strip()


def main():
    assert os.path.isdir(MODEL_DIR), f"Vosk model not found at {MODEL_DIR} (download + unzip first)"
    model = Model(MODEL_DIR)

    transcripts = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "rb") as fp:
            transcripts = pickle.load(fp)
        print(f"Resuming: {len(transcripts)} transcripts already present")

    wavs = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    print(f"{len(wavs)} wav files to consider")

    done_since_save = 0
    for fn in tqdm(wavs):
        stem = fn[:-4]
        if stem in transcripts:
            continue
        try:
            rec = KaldiRecognizer(model, 16000)
            transcripts[stem] = transcribe(rec, os.path.join(AUDIO_DIR, fn))
        except Exception as e:
            print(f"  FAIL {stem}: {type(e).__name__}: {e}")
            transcripts[stem] = ""
        done_since_save += 1
        if done_since_save >= SAVE_EVERY:
            with open(OUT_PATH, "wb") as fp:
                pickle.dump(transcripts, fp)
            done_since_save = 0

    with open(OUT_PATH, "wb") as fp:
        pickle.dump(transcripts, fp)
    n_empty = sum(1 for v in transcripts.values() if not v)
    print(f"Done. {len(transcripts)} transcripts ({n_empty} empty) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
