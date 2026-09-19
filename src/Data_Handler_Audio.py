import os
import csv
import subprocess
import tempfile

import numpy as np
import librosa

from Data_Handler import DataHandler


class AudioDataHandler(DataHandler):
    """
    Loads the COUGHVID dataset for regression on 'cough_detected'.
    Inherits split_data() / get_feature_names() from DataHandler unchanged.

    Expects a metadata CSV (csv_path) with at least columns 'uuid' and
    'cough_detected', and audio files named '<uuid>.<ext>' inside
    audio_dir. Each clip -- regardless of its original duration -- is
    converted into one small, FIXED-length feature vector (MFCC mean/std,
    zero-crossing rate, spectral centroid, RMS energy), because linear
    regression needs one fixed-size row per sample, and raw audio clips
    don't naturally have that.

    NOTE: requires ffmpeg installed and on PATH. Every clip is decoded
    through it (see _decode_to_wav) since soundfile -- librosa's default
    backend -- cannot read .webm at all, which is COUGHVID's native format.
    """

    def __init__(self, audio_dir, csv_path, label_column="cough_detected",
                 n_mfcc=13, sample_rate=16000, max_samples=None,
                 test_size=0.15, val_size=0.15, random_state=42):
        super().__init__(test_size=test_size, val_size=val_size, random_state=random_state)
        self.audio_dir = audio_dir
        self.csv_path = csv_path
        self.label_column = label_column
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        # Each clip is decoded through a real ffmpeg subprocess (~1-1.5s each),
        # so the full ~27,500-clip COUGHVID set takes hours. max_samples caps
        # how many valid rows are processed, for a tractable real-data run.
        self.max_samples = max_samples

    def _decode_to_wav(self, path):
        """Convert any input format to a temp WAV via ffmpeg, then delete the temp file.

        soundfile (librosa's default backend) cannot read .webm at all, and
        COUGHVID's recordings are distributed as .webm -- ffmpeg handles every
        format uniformly, so every clip is routed through it regardless of
        its original extension.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-nostdin", "-i", path, "-ar", str(self.sample_rate), "-ac", "1", tmp_path],
                check=True, capture_output=True, timeout=20,
            )
            signal, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
        finally:
            os.remove(tmp_path)
        return signal, sr

    def _extract_features(self, path):
        signal, sr = self._decode_to_wav(path)

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        zcr = librosa.feature.zero_crossing_rate(signal).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr).mean()
        rms = librosa.feature.rms(y=signal).mean()

        return np.concatenate([mfcc_mean, mfcc_std, [zcr, spectral_centroid, rms]])

    def _find_audio_file(self, uuid):
        for ext in (".wav", ".ogg", ".webm", ".mp3"):
            candidate = os.path.join(self.audio_dir, uuid + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def load_data(self):
        X, y = [], []
        skipped = 0

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.max_samples is not None and len(X) >= self.max_samples:
                    break

                label = row.get(self.label_column, "")
                if label in ("", None):
                    continue

                audio_path = self._find_audio_file(row["uuid"])
                if audio_path is None:
                    skipped += 1
                    continue

                try:
                    features = self._extract_features(audio_path)
                except Exception:
                    skipped += 1
                    continue

                X.append(features)
                y.append(float(label))

                if len(X) % 200 == 0:
                    print(f"AudioDataHandler: processed {len(X)} clips...", flush=True)

        if skipped:
            print(f"AudioDataHandler: skipped {skipped} rows (missing file or decode error).")

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        self.feature_names = (
            [f"mfcc_mean_{i}" for i in range(self.n_mfcc)]
            + [f"mfcc_std_{i}" for i in range(self.n_mfcc)]
            + ["zero_crossing_rate", "spectral_centroid", "rms_energy"]
        )

        return X, y
