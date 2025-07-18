import numpy as np
import struct
from array import array


def load_dataset(images_path, labels_path):
    images = load_images(images_path)
    labels = load_labels(labels_path)
    return zip(images, labels)


def load_labels(labels_path):        
    labels = []
    with open(labels_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())

    return list(map(one_hot, labels))


def load_images(images_path):
    with open(images_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())
 
    images = []
    for i in range(size):
        images.append(np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).astype(np.float32) / 255.0)
    
    return images


def one_hot(x):
    result = np.zeros(10)
    result[x] = 1
    return np.array(result)