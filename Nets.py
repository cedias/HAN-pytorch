import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict



class AttentionalBiGRU(nn.Module):

    def __init__(self, inp_size, hid_size, dropout=0):
        super(AttentionalBiGRU, self).__init__()
        self.register_buffer("mask",torch.FloatTensor())

        natt = hid_size*2
        
        self.gru = nn.GRU(input_size=inp_size,hidden_size=hid_size,num_layers=1,bias=True,batch_first=True,dropout=dropout,bidirectional=True)
        self.lin = nn.Linear(hid_size*2,natt)
        self.att_w = nn.Linear(natt,1,bias=False)
        self.tanh = nn.Tanh()


    
    def forward(self, packed_batch):
        
        rnn_sents,_ = self.gru(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        emb_h = self.tanh(self.lin(enc_sents.view(enc_sents.size(0)*enc_sents.size(1),-1)))  # Nwords * Emb

        attend = self.att_w(emb_h).view(enc_sents.size(0),enc_sents.size(1)).transpose(0,1)
        all_att = self._masked_softmax(attend,self._list_to_bytemask(list(len_s))).transpose(0,1) # attW,sent 
        attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents

        return attended.sum(0,True).squeeze(0)

    def forward_att(self, packed_batch):
        
        rnn_sents,_ = self.gru(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        
        emb_h = self.tanh(self.lin(enc_sents.view(enc_sents.size(0)*enc_sents.size(1),-1)))  # Nwords * Emb
        attend = self.att_w(emb_h).view(enc_sents.size(0),enc_sents.size(1)).transpose(0,1)
        all_att = self._masked_softmax(attend,self._list_to_bytemask(list(len_s))).transpose(0,1) # attW,sent 
        attended = all_att.unsqueeze(2).expand_as(enc_sents) * enc_sents
        return attended.sum(0,True).squeeze(0), all_att

    def _list_to_bytemask(self,l):
        mask = self._buffers['mask'].resize_(len(l),l[0]).fill_(1)

        for i,j in enumerate(l):
            if j != l[0]:
                mask[i,j:l[0]] = 0

        return mask
    
    def _masked_softmax(self,mat,mask):
        exp = torch.exp(mat) * Variable(mask,requires_grad=False)
        sum_exp = exp.sum(1,True)+0.0001
     
        return exp/sum_exp.expand_as(exp)



class HierarchicalDoc(nn.Module):

    def __init__(self, ntoken, num_class, emb_size=200, hid_size=50):
        super(HierarchicalDoc, self).__init__()

        self.embed = nn.Embedding(ntoken, emb_size,padding_idx=0)
        self.word = AttentionalBiGRU(emb_size, hid_size)
        self.sent = AttentionalBiGRU(hid_size*2, hid_size)

        self.emb_size = emb_size
        self.lin_out = nn.Linear(hid_size*2,num_class)
        self.register_buffer("reviews",torch.Tensor())


    def set_emb_tensor(self,emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor

        
    def _reorder_sent(self,sents,stats):
        
        sort_r = sorted([(l,r,s,i) for i,(l,r,s) in enumerate(stats)], key=itemgetter(0,1,2)) #(len(r),r#,s#)
        builder = OrderedDict()
        
        for (l,r,s,i) in sort_r:
            if r not in builder:
                builder[r] = [i]
            else:
                builder[r].append(i)
                
        list_r = list(reversed(builder))
        
        revs = Variable(self._buffers["reviews"].resize_(len(builder),len(builder[list_r[0]]),sents.size(1)).fill_(0), requires_grad=False)
        lens = []
        real_order = []
        
        for i,x in enumerate(list_r):
            revs[i,0:len(builder[x]),:] = sents[builder[x],:]
            lens.append(len(builder[x]))
            real_order.append(x)

        real_order = sorted(range(len(real_order)), key=lambda k: real_order[k])
        
        return revs,lens,real_order
        
    
    def forward(self, batch_reviews,stats):
        ls,lr,rn,sn = zip(*stats)
        emb_w = F.dropout(self.embed(batch_reviews),training=self.training)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)
        sent_embs = self.word(packed_sents)
        
        rev_embs,lens,real_order = self._reorder_sent(sent_embs,zip(lr,rn,sn))

        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lens,batch_first=True)
        doc_embs = self.sent(packed_rev)

        final_emb = doc_embs[real_order,:]
        out = self.lin_out(final_emb)
        
        return out


    def forward_visu(self, batch_reviews,stats):
        ls,lr,rn,sn = zip(*stats)
        emb_w = self.embed(batch_reviews)
        
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)
        sent_embs,att_w = self.word.forward_att(packed_sents)
        
        rev_embs,lens,real_order = self._reorder_sent(sent_embs,zip(lr,rn,sn))

        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lens,batch_first=True)
        doc_embs,att_s = self.sent.forward_att(packed_rev)

        final_emb = doc_embs[real_order,:]
        att_s = att_s[:,real_order]

        out = self.lin_out(final_emb)
        
        return out,att_s




