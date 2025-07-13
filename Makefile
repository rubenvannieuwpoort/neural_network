# Makefile for MNIST dataset

.PHONY: download clean


prepare: train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte


train-images-idx3-ubyte: train-images-idx3-ubyte.gz
	gunzip -k train-images-idx3-ubyte.gz

train-images-idx3-ubyte.gz:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz


train-labels-idx1-ubyte: train-labels-idx1-ubyte.gz
	gunzip -k train-labels-idx1-ubyte.gz

train-labels-idx1-ubyte.gz:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz


t10k-images-idx3-ubyte: t10k-images-idx3-ubyte.gz
	gunzip -k t10k-images-idx3-ubyte.gz

t10k-images-idx3-ubyte.gz:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz


t10k-labels-idx1-ubyte: t10k-labels-idx1-ubyte.gz
	gunzip -k t10k-labels-idx1-ubyte.gz

t10k-labels-idx1-ubyte.gz:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

clean:
	rm -f train-images-idx3-ubyte.gz train-images-idx3-ubyte train-labels-idx1-ubyte.gz train-lables-idx1-ubyte
	rm -f t10k-images-idx3-ubyte.gz t10k-images-idx3-ubyte t10k-labels-idx1-ubyte.gz t10k-lables-idx1-ubyte
