import gzip
import pickle as pkl
import argparse
from tqdm import tqdm
from random import randint
from collections import Counter
import sys
import collections
import gensim
import logging
import spacy
import itertools
import re
import json
import torch

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) #gensim logging


def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0)
    return count


def build_dataset(args):

    def preprocess(data):
        return (data['reviewText'],min(1,int(float(data["overall"])))) #zero is useless

    def preprocess_rescale(data):
        rating = data["overall"]

        if rescale:
            if rating > 3:
                rating = 1
            elif rating == 3:
                return None
            else:
                rating = 0

        return (data['reviewText'],min(1,int(float(data["overall"])))) #zero is useless

    def data_generator(data):
        with gzip.open(args.input,"r") as f:
            for x in tqdm(f,desc="Reviews",total=count_lines(f)):
                yield json.loads(x)

    class TokIt(collections.Iterator):
        def __init__(self, tokenized):
            self.tok = tokenized
            self.x = 0
            self.stop = len(tokenized)

        def __iter__(self):
            return self

        def next(self):
            if self.x < self.stop:
                self.x += 1
                return list(w.orth_ for w in self.tok[self.x-1])
            else:
                self.x = 0
                raise StopIteration 
        __next__ = next


    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    nlp = spacy.load('en')

    tokenized = [tok for tok in tqdm(nlp.tokenizer.pipe((x["reviewText"] for x in data_generator(args.input)),batch_size=10000, n_threads=8),desc="Tokenizing")]

    if args.create_emb:
        w2vmodel = gensim.models.Word2Vec(TokIt(tokenized), size=args.emb_size, window=5,min_count=0,iter=args.epochs, max_vocab_size=args.dic_size, workers=4)
        print(len(w2vmodel.wv.vocab))
        w2vmodel.wv.save_word2vec_format(args.emb_file,total_vec=len(w2vmodel.wv.vocab))        

    if args.rescale:
        print("-> Rescaling data to 0-1 (3's are discarded)")
        data = [preprocess_rescale(dt) for dt,tok in tqdm(zip(data_generator(args.input),tokenized),desc="Processing")]
        data = [d for d in tqdm(data,desc="Removing Nones, 3's") if d is not None]

    else:
        data = [preprocess(dt) for dt,tok in tqdm(zip(data_generator(args.input),tokenized),desc="Processing")]


    splits = [randint(0,args.nb_splits-1) for _ in range(0,len(data))]
    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    return {"data":data,"splits":splits,"rows":("review","rating")}


def main(args):
    ds = build_dataset(args)
    pkl.dump(ds,open(args.output,"wb"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="sentences.pkl")
    parser.add_argument("--rescale",action="store_true")
    parser.add_argument("--nb_splits",type=int, default=5)

    parser.add_argument("--create-emb",action="store_true")
    parser.add_argument("--emb-file", type=str, default=None)
    parser.add_argument("--emb-size",type=int, default=100)
    parser.add_argument("--dic-size", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=10)
    args = parser.parse_args()

    if args.emb_file is None:
        args.emb_file = args.output + "_emb.txt"

    main(args)