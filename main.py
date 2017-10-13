
import sys
import argparse
import torch
import spacy
import pickle as pkl
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from collections import Counter
from tqdm import tqdm
from operator import itemgetter
from random import choice
from collections import OrderedDict,Counter
from Nets import HierarchicalDoc
from Data import TuplesListDataset, Vectorizer, BucketSampler




def checkpoint(epoch,net,output):
    model_out_path = output+"_epoch_{}.pth".format(epoch)
    torch.save(net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def load_embeddings(file):
    emb_file = open(file).readlines()
    first = emb_file[0]
    print(first)
    word, vec = int(first.split()[0]),int(first.split()[1])
    size = (word,vec)
    print("--> Got {} words of {} dimensions".format(size[0],size[1]))
    tensor = np.zeros((size[0]+2,size[1]),dtype=np.float32) ## adding padding + unknown
    word_d = {}
    word_d["_padding_"] = 0
    word_d["_unk_word_"] = 1

    print(tensor.shape)

    for i,line in tqdm(enumerate(emb_file,1),desc="Creating embedding tensor",total=len(emb_file)):
        if i==1: #skipping header (-1 to the enumeration to take it into account)
            continue

        spl = line.strip().split(" ")

        if len(spl[1:]) == size[1]: #word is most probably whitespace or junk if badly parsed
            word_d[spl[0]] = i
            tensor[i] = np.array(spl[1:],dtype=np.float32)
            


    return tensor, word_d

def tuple_batcher_builder(vectorizer, train=True):

    def tuple_batch(l):
        review,rating = zip(*l)
        r_t = torch.max(torch.zeros(1).long(),torch.Tensor(rating).long() - 1)
        list_rev = vectorizer.vectorize_batch(review,train)

        # sorting by sentence-review length
        stat =  sorted([(len(s),len(r),r_n,s_n,s) for r_n,r in enumerate(list_rev) for s_n,s in enumerate(r)],reverse=True)

        max_len = stat[0][0]
        batch_t = torch.zeros(len(stat),max_len).long()

        for i,s in enumerate(stat):
            for j,w in enumerate(s[-1]): # s[-1] is sentence in stat tuple
                batch_t[i,j] = w
                
        stat = [(ls,lr,r_n,s_n) for ls,lr,r_n,s_n,_ in stat]
        
        return batch_t,r_t, stat,review

    return tuple_batch


def tuple2var(tensors,data):
    def copy2tensor(t,data):
        t.resize_(data.size()).copy_(data)
        return Variable(t)
    return tuple(map(copy2tensor,tensors,data))


def new_tensors(n,cuda,types={}):
    def new_tensor(t_type,cuda):
        x = torch.Tensor()

        if t_type:
            x = x.type(t_type)
        if cuda:
            x = x.cuda()
        return x

    return tuple([new_tensor(types.setdefault(i,None),cuda) for i in range(0,n)])

def train(epoch,net,optimizer,dataset,criterion,cuda):
    epoch_loss = 0
    ok_all = 0
    data_tensors = new_tensors(2,cuda,types={0:torch.LongTensor,1:torch.LongTensor}) #data-tensors

    with tqdm(total=len(dataset),desc="Training") as pbar:
        for iteration, (batch_t,r_t,stat,rev) in enumerate(dataset):
            
            data = tuple2var(data_tensors,(batch_t,r_t))
            optimizer.zero_grad()
            out = net(data[0],stat)
            ok,per = accuracy(out,data[1])
            loss = criterion(out, data[1])
            epoch_loss += loss.data[0]
            loss.backward()

            optimizer.step()

            ok_all += per.data[0]

            
            

            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/(iteration+1),"CE":epoch_loss/(iteration+1)})

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, {}% accuracy".format(epoch, epoch_loss /len(dataset),ok_all/len(dataset)))



def test(epoch,net,dataset,cuda):
    epoch_loss = 0
    ok_all = 0
    pred = 0
    skipped = 0
    data_tensors = new_tensors(2,cuda,types={0:torch.LongTensor,1:torch.LongTensor}) #data-tensors
    with tqdm(total=len(dataset),desc="Evaluating") as pbar:
        for iteration, (batch_t,r_t, stat,rev) in enumerate(dataset):
            data = tuple2var(data_tensors,(batch_t,r_t))
            out,att = net.forward_visu(data[0],stat)

            ok,per = accuracy(out,data[1])
            ok_all += per.data[0]
            pred+=1


            
            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/pred, "skipped":skipped})


    print("===> TEST Complete:  {}% accuracy".format(ok_all/pred))



def accuracy(out,truth):
    def sm(mat):
        exp = torch.exp(mat)
        sum_exp = exp.sum(1,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    _,max_i = torch.max(sm(out),1)


    eq = torch.eq(max_i,truth).float()
    all_eq = torch.sum(eq)

    return all_eq, all_eq/truth.size(0)*100

def main(args):
    print(32*"-"+"\nHierarchical Attention Network:\n" + 32*"-")
    print("\nLoading Data:\n" + 25*"-")

    max_features = args.max_feat
    datadict = pkl.load(open(args.filename,"rb"))
    tuples = datadict["data"]
    splits  = datadict["splits"]
    split_keys = set(x for x in splits)

    if args.split not in split_keys:
        print("Chosen split (#{}) not in split set {}".format(args.split,split_keys))
    else:
        print("Split #{} chosen".format(args.split))

    train_set,test_set = TuplesListDataset.build_train_test(tuples,splits,args.split)

    print("Train set length:",len(train_set))
    print("Test set length:",len(test_set))

    classes = train_set.get_class_dict(1)
    num_class = len(classes)

    
    print(25*"-"+"\nClass stats:\n" + 25*"-")
    print("Train set:\n" + 10*"-")

    class_stats,class_per = train_set.get_stats(1)
    print(class_stats)
    print(class_per)

    print(10*"-" + "\n Test set:\n" + 10*"-")
    
    test_stats,test_per = test_set.get_stats(1)
    print(test_stats) 
    print(test_per)


    print(25*"-" + "\nBuilding word vectors: \n"+"-"*25)

    vectorizer = Vectorizer(max_word_len=args.max_words,max_sent_len=args.max_sents)

    if args.emb:
        tensor,dic = load_embeddings(args.emb)
        net = HierarchicalDoc(ntoken=len(dic),emb_size=len(tensor[1]),num_class=num_class)
        net.set_emb_tensor(torch.FloatTensor(tensor))
        vectorizer.word_dict = dic
    else:
        vectorizer.build_dict(train_set.field_iter(0),args.max_feat)
        net = HierarchicalDoc(ntoken=len(vectorizer.word_dict),num_class=num_class)


    tuple_batch = tuple_batcher_builder(vectorizer,train=True)
    tuple_batch_test = tuple_batcher_builder(vectorizer,train=False)
 
    

    sampler = None
    if args.balance:
        sampler =  BucketSampler(train_set)
        sampler_t = BucketSampler(test_set)


        dataloader = DataLoader(train_set, batch_size=args.b_size, shuffle=False, sampler=sampler, num_workers=2, collate_fn=tuple_batch,pin_memory=True)
        dataloader_test = DataLoader(test_set, batch_size=args.b_size, shuffle=False,  num_workers=2, collate_fn=tuple_batch_test)
    else:
        dataloader = DataLoader(train_set, batch_size=args.b_size, shuffle=True, num_workers=2, collate_fn=tuple_batch,pin_memory=True)
        dataloader_test = DataLoader(test_set, batch_size=args.b_size, shuffle=True, num_workers=2, collate_fn=tuple_batch_test)


    criterion = torch.nn.CrossEntropyLoss()
      


    if args.cuda:
        net.cuda()
    
    print("-"*20)


    optimizer = optim.SGD(net.parameters(),lr=args.lr)
    torch.nn.utils.clip_grad_norm(net.parameters(), 0.5)


    for epoch in range(1, args.epochs + 1):
        for _ in range(10):
            train(epoch,net,optimizer,dataloader,criterion,args.cuda)
        test(epoch,net,dataloader_test,args.cuda)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Attention Networks for Document Classification')
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--b-size", type=int, default=32)
    parser.add_argument("--max-feat", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=10)
    parser.add_argument("--clip-grad", type=float,default=0.25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-words", type=int,default=32)
    parser.add_argument("--max-sents",type=int,default=8)
    parser.add_argument("--emb", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--balance', action='store_true',
                        help='balance class in batches')
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    main(args)
