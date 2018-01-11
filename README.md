-----------------

## Deprecated code
A faster and up to date implementation is [in my other repo](https://github.com/cedias/Hierarchical-Sentiment)

----------------

# HAN-pytorch
Batched implementation of [Hierarchical Attention Networks for Document Classification paper](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

## Requirements
- Pytorch (>= 0.2)
- Spacy (for tokenizing)
- Gensim (for building word vectors)
- tqdm (for fancy graphics)

## Scripts:
- `prepare_data.py` transforms gzip files as found on [Julian McAuley Amazon product data page](http://jmcauley.ucsd.edu/data/amazon/) to lists of `(user,item,review,rating)` tuples and builds word vectors if `--create-emb` option is specified.
- `main.py` trains a Hierarchical Model.
- `Data.py` holds data managing objects.
- `Nets.py` holds networks.
- `beer2json.py` is an helper script if you happen to have the ratebeer/beeradvocate datasets.

## Note:
The whole dataset is used to create word embeddings which can be an issue.
