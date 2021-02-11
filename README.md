# LSTM+Transformer Text Generation

<!-- Please read [this article]() (in Japanese). -->

This repository is based on [Word Language Model in pytorch/examples](https://github.com/pytorch/examples/tree/490243127c02a5ea3348fa4981ecd7e9bcf6144c/word_language_model).

## Installation

### For Pip

```
pip install -r requirements.txt
```

### For Pipenv

Python 3.8 is required.

```
pipenv install
```

### Note

If you use TensorBoard, [install TensorFlow](https://www.tensorflow.org/install/pip).


## Usage

Preparing Wikitext-103 corpus

```
./download_wikitext_103.sh
```

Training

```
./train.sh [ LSTMTransformer, Transformer, LSTM ]

// You can use TensorBoard
tensorboard --logdir=./logs
```

Generating text

```
./generate.sh [ LSTMTransformer, Transformer, LSTM ]
```

## License

See [LICENSE](https://github.com/tanikawa04/lstm-transformer/blob/master/LICENSE).
