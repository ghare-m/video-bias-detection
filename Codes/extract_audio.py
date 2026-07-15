"""extract_audio.py  (NEW — not part of the original HateMM repo)

The original scripts read $HATEMM_ROOT/AudioFiles/<stem>.wav but the repo ships no code
to create them. This script extracts the audio track of every video into a 16 kHz mono
PCM wav — a format that is both Vosk-ready (for transcription) and fine for librosa MFCC.

Input  : $HATEMM_ROOT/AllVideos/*.mp4
Output : $HATEMM_ROOT/AudioFiles/<stem>.wav
         $HATEMM_ROOT/audio_extract_failures.txt   (videos with no/!decodable audio)

moviepy is 2.x in this env -> `from moviepy import VideoFileClip` (NOT moviepy.editor).
"""

import os
import glob
import warnings

from tqdm import tqdm
from moviepy import VideoFileClip

ROOT = os.environ.get("HATEMM_ROOT", "/home/gharem/Work/Dissertation/HateMM/data")
VIDEO_DIR = os.path.join(ROOT, "AllVideos")
OUT_DIR = os.path.join(ROOT, "AudioFiles")
FAIL_LOG = os.path.join(ROOT, "audio_extract_failures.txt")

TARGET_SR = 16000  # Vosk wants 16 kHz mono


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")

    failures = []
    for path in tqdm(videos):
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(OUT_DIR, stem + ".wav")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue  # resumable
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clip = VideoFileClip(path)
                if clip.audio is None:
                    failures.append(f"{stem}\tno-audio-track")
                    clip.close()
                    continue
                # 16 kHz, 16-bit PCM, mono (-ac 1) -> Vosk-ready + librosa-friendly
                clip.audio.write_audiofile(
                    out, fps=TARGET_SR, nbytes=2, codec="pcm_s16le",
                    ffmpeg_params=["-ac", "1"], logger=None,
                )
                clip.close()
        except Exception as e:
            failures.append(f"{stem}\t{type(e).__name__}: {e}")

    n_done = len([f for f in os.listdir(OUT_DIR) if f.endswith(".wav")])
    print(f"Done. {n_done} wav files in {OUT_DIR}; {len(failures)} failures.")
    if failures:
        with open(FAIL_LOG, "w", encoding="utf-8") as fp:
            fp.write("\n".join(failures) + "\n")
        print(f"Failure list -> {FAIL_LOG}")


if __name__ == "__main__":
    main()
