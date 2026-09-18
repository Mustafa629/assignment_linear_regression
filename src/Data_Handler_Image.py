import os
import csv

import numpy as np
from PIL import Image

from Data_Handler import DataHandler


class ImageDataHandler(DataHandler):
    """
    Loads a face-image dataset for AGE REGRESSION. Inherits
    split_data() / get_feature_names() from DataHandler unchanged -- only
    load_data() differs, since only the SOURCE of (X, y) changes, not what
    happens to them afterward.

    RECOMMENDED DATASET: Kaggle's "frabbisw/facial-age"
    (https://www.kaggle.com/datasets/frabbisw/facial-age), NOT the
    "mariafrenti/age-prediction" dataset the assignment sheet suggests
    first. Both are acceptable per the assignment ("you can visit Kaggle
    to find a dataset yourself too") -- facial-age is used here because
    its folder-per-age layout is independently confirmed (cited as
    reference [30] in a related DIU thesis on the same task), whereas
    mariafrenti's exact internal layout was never directly inspected in
    this session (no internet access here to check it). Download
    facial-age, extract it, and point root_dir at the extracted folder.

    Expected layouts (auto-detected):
      1. Folder-per-age (facial-age's actual layout): root_dir contains
         subfolders named by an integer age (e.g. root_dir/023/*.jpg),
         one folder per age value.
      2. CSV metadata (fallback, for other datasets): a CSV file
         (csv_path) with columns 'filename' and 'age', where 'filename'
         is a path relative to root_dir.

    Images are converted to grayscale and resized to a fixed
    (image_size, image_size) square, then flattened into a 1D vector per
    image -- this turns each image into exactly the same "row of numbers"
    shape (d,) that every other dataset produces, so DataPreprocessor,
    LinearRegressionModel, etc. need no changes at all.
    """

    def __init__(self, root_dir, csv_path=None, image_size=32,
                 test_size=0.15, val_size=0.15, random_state=42):
        super().__init__(test_size=test_size, val_size=val_size, random_state=random_state)
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.image_size = image_size

    def _load_image(self, path):
        img = Image.open(path).convert("L")  # grayscale
        img = img.resize((self.image_size, self.image_size))
        return np.asarray(img, dtype=np.float64).flatten()

    def _load_from_csv(self):
        X, y = [], []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = os.path.join(self.root_dir, row["filename"])
                X.append(self._load_image(image_path))
                y.append(float(row["age"]))
        return np.array(X), np.array(y).reshape(-1, 1)

    def _load_from_folders(self):
        X, y = [], []
        for entry in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, entry)
            if not os.path.isdir(folder_path):
                continue
            try:
                age = float(entry)
            except ValueError:
                continue  # skip non-numeric folder names

            for filename in sorted(os.listdir(folder_path)):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                X.append(self._load_image(os.path.join(folder_path, filename)))
                y.append(age)
        return np.array(X), np.array(y).reshape(-1, 1)

    def load_data(self):
        if self.csv_path is not None:
            X, y = self._load_from_csv()
        else:
            X, y = self._load_from_folders()

        n_pixels = self.image_size * self.image_size
        self.feature_names = [f"pixel_{i}" for i in range(n_pixels)]

        return X, y
