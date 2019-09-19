import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from tqdm import tqdm
from collections import Counter

use_cuda = torch.cuda.is_available()

'''
Function to batchify the dataset for faster processing. To make sure each batch has a
sequence length of the longest sentence in that batch.
Input : Batch of varying length sequences
Output : Tensor containing a hextuple <u1, u1_length, u2, u2_length, u3, u3_length>
'''
def batchify(batch, retrieve=True, slots=False):
    vocab = {}
    with open("data/vocab.txt", 'r') as f:
            tokens = [line.strip() for line in f.readlines()]

    for i, token in enumerate(tokens):
        vocab[token] = i
    # input is a list of Triple objects


    bt_siz = len(batch)
    # sequence length only affects the memory requirement, otherwise longer is better
    pad_idx, max_seq_len = 9182, 160

    u1_batch, u2_batch, u3_batch = [], [], []
    u1_lens, u2_lens, u3_lens = np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int)

    if retrieve==True:
        u_ret_batch = []
        uret_lens = np.zeros(bt_siz, dtype=int)


    # these store the max sequence lengths for the batch
    if retrieve == False:
        l_u1, l_u2, l_u3 = 0, 0, 0
        for i, (d, cl_u1, cl_u2, cl_u3) in enumerate(batch):
            cl_u1 = min(cl_u1, max_seq_len)
            cl_u2 = min(cl_u2, max_seq_len)
            cl_u3 = min(cl_u3, max_seq_len)

            if cl_u1 > l_u1:
                l_u1 = cl_u1

            utt_1 = [vocab[word] for word in d[0].split()]
            u1_batch.append(torch.LongTensor(utt_1))

            u1_lens[i] = cl_u1

            if cl_u2 > l_u2:
                l_u2 = cl_u2
            utt_2 = [vocab[word] for word in d[1].split()]
            u2_batch.append(torch.LongTensor(utt_2))
            u2_lens[i] = cl_u2

            if cl_u3 > l_u3:
                l_u3 = cl_u3
            utt_3 = [vocab[word] for word in d[2].split()]
            u3_batch.append(torch.LongTensor(utt_3))
            u3_lens[i] = cl_u3

    if retrieve==True:
        l_ret = 0
        
        l_u1, l_u2, l_u3 = 0, 0, 0
        for i, (d, cl_u1, cl_u2, cl_u3, d2, _, _,cl_ret ) in enumerate(batch):
            cl_u1 = min(cl_u1, max_seq_len)
            cl_u2 = min(cl_u2, max_seq_len)
            cl_u3 = min(cl_u3, max_seq_len)

            if cl_u1 > l_u1:
                l_u1 = cl_u1

            utt_1 = [vocab[word] for word in d[0].split()]
            u1_batch.append(torch.LongTensor(utt_1))

            u1_lens[i] = cl_u1

            if cl_u2 > l_u2:
                l_u2 = cl_u2
            utt_2 = [vocab[word] for word in d[1].split()]
            u2_batch.append(torch.LongTensor(utt_2))
            u2_lens[i] = cl_u2

            if cl_u3 > l_u3:
                l_u3 = cl_u3
            utt_3 = [vocab[word] for word in d[2].split()]
            u3_batch.append(torch.LongTensor(utt_3))
            u3_lens[i] = cl_u3
            
            
            cl_ret= min(cl_ret, max_seq_len)

            if cl_ret > l_ret:
                l_ret = cl_ret

            ret = [vocab[word] for word in d2[-1].split()]
       
            u_ret_batch.append(torch.LongTensor(ret))

            uret_lens[i] = cl_ret

    t1, t2, t3 = u1_batch, u2_batch, u3_batch

    if retrieve==True:
        t_ret = u_ret_batch
        u_ret_batch = Variable(torch.ones(bt_siz, l_ret).long() * pad_idx)

    u1_batch = Variable(torch.ones(bt_siz, l_u1).long() * pad_idx)
    u2_batch = Variable(torch.ones(bt_siz, l_u2).long() * pad_idx)
    u3_batch = Variable(torch.ones(bt_siz, l_u3).long() * pad_idx)
    end_tok = torch.LongTensor(vocab["<eou>"])

    for i in range(bt_siz):
      
        seq1, cur1_l = t1[i], t1[i].size(0)
        if cur1_l <= l_u1:
            u1_batch[i, :cur1_l].data.copy_(seq1[:cur1_l])
        else:
            u1_batch[i, :].data.copy_(torch.cat((seq1[:l_u1-1], end_tok), 0))

        seq2, cur2_l = t2[i], t2[i].size(0)
        if cur2_l <= l_u2:
            u2_batch[i, :cur2_l].data.copy_(seq2[:cur2_l])
        else:
            u2_batch[i, :].data.copy_(torch.cat((seq2[:l_u2-1], end_tok), 0))

        seq3, cur3_l = t3[i], t3[i].size(0)
        if cur3_l <= l_u3:
            u3_batch[i, :cur3_l].data.copy_(seq3[:cur3_l])
        else:
            u3_batch[i, :].data.copy_(torch.cat((seq3[:l_u3-1], end_tok), 0))

        if retrieve==True:
            ret, ret_l = t_ret[i], t_ret[i].size(0)
            if ret_l <= l_ret:
                u_ret_batch[i, :ret_l].data.copy_(ret[:ret_l])
            else:
                u_ret_batch[i, :].data.copy_(torch.cat((ret[:l_ret-1], end_tok), 0))

    sort1, sort2, sort3 = np.argsort(u1_lens*-1), np.argsort(u2_lens*-1), np.argsort(u3_lens*-1)
    # cant call use_cuda here because this function block is used in threading calls
    if retrieve==True:
        sort_ret = np.argsort(uret_lens*-1)
        return u1_batch[sort1, :], u1_lens[sort1], u2_batch[sort2, :], u2_lens[sort2], u3_batch[sort3, :], u3_lens[sort3],  u_ret_batch[sort_ret, :], uret_lens[sort_ret]

    
    return u1_batch[sort1, :], u1_lens[sort1], u2_batch[sort2, :], u2_lens[sort2], u3_batch[sort3, :], u3_lens[sort3]



'''
Function to convert tensor of sequnces of token ids to sentences
'''
def id_to_sentence(x, inv_dict, id_eou=4105, greedy=False):
    sents = []
    inv_dict[9182] = '<pad>'
    for li in x:
        if not greedy:
            scr = li[1]
            seq = li[0]
        else:
            scr = 0
            seq = li
        sent = []
       
        for i in seq:

            if type(i) != int:
                sent.append(inv_dict[i.item()])
                if i.item() == id_eou:
                    break
            else:
                sent.append(inv_dict[i])
                if i == id_eou:
                    break
            
        sents.append((" ".join(sent), scr))
    return sents


'''
Function to initialize all the parameters of the model.
According to the paper, rnn weights are orthogonally initialized, all other parameters are from a gaussian distribution with 0 mean and sd of 0.01.
'''
def init_param(model):
    for name, param in model.named_parameters():
        # skip over the embeddings so that the padding index ones are 0
        if 'embed' in name:
            continue
        elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)

'''
Function to calculate the liklihood of a given output sentence
'''
def get_sent_ll(u3, u3_lens, model, criteria, ses_encoding):
    preds, _ = model.dec([ses_encoding, u3, u3_lens])
    preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
    u3 = u3[:, 1:].contiguous().view(-1)
    loss = criteria(preds, u3).item()
    target_tokens = u3.ne(9182).long().sum().item()
    return -1*loss/target_tokens


def calc_valid_loss(data_loader, criteria, model):
    model.eval()
    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    # we want to find the perplexity or likelihood of the provided sequence

    valid_loss, num_words = 0, 0
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        preds, lmpreds = model(sample_batch)
        u3 = sample_batch[4]
        if use_cuda:
            u3 = u3.cuda()
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        u3 = u3[:, 1:].contiguous().view(-1)
        # do not include the lM loss, exp(loss) is perplexity
        loss = criteria(preds, u3)
        num_words += u3.ne(9182).long().sum().item()
        valid_loss += loss.item()

    model.train()
    model.dec.set_teacher_forcing(cur_tc)

    return valid_loss/num_words


def uniq_answer(fil):
    uniq = Counter()
    with open(fil + '_result.txt', 'r') as fp:
        all_lines=  fp.readlines()
        for line in all_lines:
            resp = line.split("    |    ")
            uniq[resp[1].strip()] += 1
    print('uniq', len(uniq), 'from', len(all_lines))
    print('---all---')
    for s in uniq.most_common():
        print(s)


def sort_key(temp, mmi):
    if mmi:
        lambda_param = 0.25
        return temp[1] - lambda_param*temp[2] + len(temp[0])*0.1
    else:
        return temp[1]/len(temp[0])**0.7

def max_out(x):
    # make sure s2 is even and that the input is 2 dimension
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x