import os
import tempfile
import gzip
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import struct
from array import array
from dataclasses import dataclass


@dataclass
class Dataset:
    Path: str
    URL: str


TRAINING_IMAGES = Dataset('./data/mnist/train-images.idx3-ubyte', 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz')
TRAINING_LABELS = Dataset('./data/mnist/train-labels-idx1-ubyte', 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz')
TEST_IMAGES = Dataset('./data/mnist/t10k-images.idx3-ubyte', 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz')
TEST_LABELS = Dataset('./data/mnist/t10k-labels-idx1-ubyte', 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz')


class MNIST:
    def training_set():
        return load_dataset(TRAINING_IMAGES, TRAINING_LABELS)

    def test_set():
        return load_dataset(TEST_IMAGES, TEST_LABELS)


def load_dataset(image_dataset: Dataset, label_data: Dataset):
    load_to_disk(image_dataset)
    images = load_images(image_dataset.Path)

    load_to_disk(label_data)
    labels = load_labels(label_data.Path)

    return list(zip(images, labels))


def load_to_disk(dataset: Dataset):
    if os.path.exists(dataset.Path):
        return

    with tempfile.NamedTemporaryFile(delete=False) as tmp_download:
        with urllib.request.urlopen(dataset.URL) as response:
            shutil.copyfileobj(response, tmp_download)
        tmp_download_path = tmp_download.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp_extracted:
        with gzip.open(tmp_download_path, 'rb') as f_in:
            shutil.copyfileobj(f_in, tmp_extracted)
        tmp_extracted_path = tmp_extracted.name

    destination = Path(dataset.Path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp_extracted_path, dataset.Path)
    Path(tmp_download_path).unlink(missing_ok=True)


def load_labels(labels_path):        
    labels = []
    with open(labels_path, 'rb') as file:
        magic, _ = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())

    return labels  # TODO: or list(labels)?


def load_images(images_path):
    with open(images_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())
 
    images = []
    for i in range(size):
        images.append(np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).astype(np.float32).reshape(1, -1) / 255.0)
    
    return images
