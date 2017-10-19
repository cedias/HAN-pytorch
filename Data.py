import spacy
import random
import pickle as pkl
import numpy as np
import torch
import itertools

import torch.utils.data as data
import torch.nn.functional as fn

from torch.autograd import Variable
from collections import Counter
from tqdm import tqdm
from itertools import groupby
from operator import itemgetter
from collections import OrderedDict
import pickle as pkl
from random import choice, random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler



class TuplesListDataset(Dataset):

    def __init__(self, tuplelist):
        super(TuplesListDataset, self).__init__()
        self.tuplelist = tuplelist

    def __len__(self):
        return len(self.tuplelist)

    def __getitem__(self,index):
        return self.tuplelist[index]

    def field_iter(self,field):
        def field_iterator():
            for x in self.tuplelist:
                yield x[field]

        return field_iterator


    def get_stats(self,field):
        d =  dict(Counter(self.field_iter(field)()))
        sumv = sum([v for k,v in d.items()])
        class_per = {k:(v/sumv) for k,v  in d.items()}

        return d,class_per

    def get_class_dict(self,field):
        self.class_mapping = {i:c for i,c in  enumerate(sorted(list(set(self.field_iter(field)()))))}
        return self.class_mapping

    @staticmethod
    def build_train_test(datatuples,splits,split_num=0):
        train,test = [],[]

        for split,data in tqdm(zip(splits,datatuples),total=len(datatuples),desc="Building train/test of split #{}".format(split_num)):
            if split == split_num:
                test.append(data)
            else:
                train.append(data)
        return TuplesListDataset(train),TuplesListDataset(test)
        


class BucketSampler(Sampler):
    """
    Evenly sample from bucket for datalen
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.index_buckets = self._build_index_buckets()
        self.len = min([len(x) for x in self.index_buckets.values()])

    def __iter__(self):

        return iter(self.bucket_iterator())

    def __len__(self):

        return self.len

    def bucket_iterator(self):
        cl = list(self.index_buckets.keys())
   
        for x in range(len(self)):
            yield choice(self.index_buckets[choice(cl)])

            
    def _build_index_buckets(self):
        class_index = {}
        for ind,cl in enumerate(self.dataset.field_iter(1)()):
            if cl not in class_index:
                class_index[cl] = [ind]
            else:
                class_index[cl].append(ind)
        return class_index
        



class Vectorizer():

    def __init__(self,word_dict=None,max_sent_len=8,max_word_len=32):
        self.word_dict = word_dict
        self.nlp = spacy.load('en')
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len


    def _get_words_dict(self,data,max_words):
        word_counter = Counter(w.lower_ for w in self.nlp.tokenizer.pipe((doc for doc in data())))
        dict_w["_padding_"] = 0
        dict_w["_unk_word_"] = 1
        print("Dictionnary has {} words".format(len(dict_w)))
        return dict_w

    def build_dict(self,text_iterator,max_f):
        self.word_dict = self._get_words_dict(text_iterator,max_f)

    def vectorize_batch(self,t,trim=True):
        return self._vect_dict(t,trim)

    def _vect_dict(self,t,trim):

        if self.word_dict is None:
            print("No dictionnary to vectorize text \n-> call method build_dict \n-> or set a word_dict attribute \n first")
            raise Exception

        revs = []
        for rev in t:
            review = []
            for j,sent in enumerate(self.nlp(rev).sents):  

                if trim and j>= self.max_sent_len:
                    break
                s = []
                for k,word in enumerate(sent):
                    word = word.lower_

                    if trim and k >= self.max_word_len:
                        break

                    if word in self.word_dict:
                        s.append(self.word_dict[word])
                    else:
                        s.append(self.word_dict["_unk_word_"]) #_unk_word_
                 
                if len(s) >= 1:
                    review.append(torch.LongTensor(s))
            if len(review) == 0:
                review = [torch.LongTensor([self.word_dict["_unk_word_"]])]        
            revs.append(review)

        return revs

