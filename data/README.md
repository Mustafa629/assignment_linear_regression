# Datasets

## Numeric (already handled automatically)
`sklearn.datasets.fetch_california_housing()` downloads and caches itself — nothing to
place here.

## Image — face age regression
Recommended: **Kaggle `frabbisw/facial-age`**
https://www.kaggle.com/datasets/frabbisw/facial-age

1. `kaggle datasets download -d frabbisw/facial-age` (needs your Kaggle API token at
   `~/.kaggle/kaggle.json`), then unzip into `data/facial-age/`.
2. Confirm the layout: folders named by integer age (e.g. `data/facial-age/023/*.jpg`).
3. Run:
   ```python
   from Data_Handler_Image import ImageDataHandler
   dh = ImageDataHandler(root_dir="data/facial-age", image_size=32)
   ```

## Audio — COUGHVID cough_detected regression
https://zenodo.org/records/4498364 → download `public_dataset.zip`.

1. Extract into `data/coughvid/` — expect a `metadata_compiled.csv` (or similar) with a
   `uuid` column and a `cough_detected` column, plus one audio file per uuid.
2. If files are `.webm`, install `ffmpeg` and ensure it's on `PATH` (librosa delegates
   decoding to it).
3. Run:
   ```python
   from Data_Handler_Audio import AudioDataHandler
   dh = AudioDataHandler(audio_dir="data/coughvid", csv_path="data/coughvid/metadata_compiled.csv")
   ```
4. If the real CSV's column names differ from `uuid`/`cough_detected`, adjust the
   `label_column` argument or the CSV reading in `AudioDataHandler.load_data()`.

Neither dataset could be downloaded automatically in the environment this project was
built in (no reliable outbound internet, plus Kaggle requires your own account
credentials) — both handlers were instead verified end-to-end on small synthetic
stand-in data. Once you download the real files here, no other code changes should be
needed.
