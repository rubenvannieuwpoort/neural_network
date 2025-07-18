# Neural network

This is an extremely simple implementation of a neural network in Python. It takes about 75 lines of code.

It uses fully connected linear layers and a ReLU activation function, and uses the mean square error loss function. It trains on the MNIST dataset and reaches about 95% accuracy.


## Running

You probably want to set up a virtual environment:
```
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
```

Then, make sure the MNIST dataset is available. This is most easily done with the provided `Makefile`:
```
$ make
```

Now, we can run `main.py`. I usually use it to output a CSV file:
```
$ python main.py > output.csv
```

Now, you can open `output.csv` in Excel or OpenOffice Calc, and make a pretty graph of the accuracy.


## TODO
- batching
- Kaiment initialization of the weights
- softmax layer
- cross-entropy loss
- different optimizers
- dropout
- l2 regularization
