import torch
from torchvision.datasets import VisionDataset
import numpy as np
import urllib
from os import path, makedirs


class MovingMNIST(VisionDataset):
    mirrors = [
        "https://github.com/eadali/moving-mnist/releases/download/v0.1/",
    ]

    resources = [
        ("train-sequences.npy", ""),
        ("train-annotations.npy", ""),
    ]

    def __init__(self, root, download=False):
        self.root = root
        self.seq_path = path.join(self.root, 'train-sequences.npy')
        self.ann_path = path.join(self.root, 'train-annotations.npy')

        if download:
            self.download()

        self.sequences = torch.from_numpy(np.load(self.seq_path))
        self.annotations = torch.from_numpy(np.load(self.ann_path))

    def __len__(self):
        return self.sequences.shape[1]

    def __getitem__(self, idx):
        return self.sequences[idx], self.annotations[idx]

    def download(self):
        if not path.exists(self.root):
            makedirs(self.root)

        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                file_path = path.join(self.root, filename)
                try:
                    print(f"Downloading {url}")
                    urllib.request.urlretrieve(url, file_path)
                except urllib.error.URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")