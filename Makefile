# Makefile for MNIST dataset

.PHONY: download clean


prepare: data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte


data/mnist/train-images-idx3-ubyte: data/download/train-images-idx3-ubyte.gz
	mkdir -p data/mnist
	gunzip -kc data/download/train-images-idx3-ubyte.gz > data/mnist/train-images-idx3-ubyte

data/download/train-images-idx3-ubyte.gz:
	mkdir -p data/download
	wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz -O data/download/train-images-idx3-ubyte.gz


data/mnist/train-labels-idx1-ubyte: data/download/train-labels-idx1-ubyte.gz
	mkdir -p data/mnist
	gunzip -kc data/download/train-labels-idx1-ubyte.gz > data/mnist/train-labels-idx1-ubyte

data/download/train-labels-idx1-ubyte.gz:
	mkdir -p data/download
	wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz -O data/download/train-labels-idx1-ubyte.gz


data/mnist/t10k-images-idx3-ubyte: data/download/t10k-images-idx3-ubyte.gz
	mkdir -p data/mnist
	gunzip -kc data/download/t10k-images-idx3-ubyte.gz > data/mnist/t10k-images-idx3-ubyte

data/download/t10k-images-idx3-ubyte.gz:
	mkdir -p data/download
	wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz -O data/download/t10k-images-idx3-ubyte.gz


data/mnist/t10k-labels-idx1-ubyte: data/download/t10k-labels-idx1-ubyte.gz
	mkdir -p data/mnist
	gunzip -kc data/download/t10k-labels-idx1-ubyte.gz > data/mnist/t10k-labels-idx1-ubyte

data/download/t10k-labels-idx1-ubyte.gz:
	mkdir -p data/download
	wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz -O data/download/t10k-labels-idx1-ubyte.gz

clean:
	rm -rf data
