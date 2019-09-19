import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import numpy as np
from model.util import *

use_cuda = torch.cuda.is_available()






# encode each sentence utterance into a single vector
class UtteranceEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, options):
        super(UtteranceEncoder, self).__init__()
        self.use_embed = options.use_embed
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(options.drp)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=9182, sparse=False)
        if self.use_embed:
            pretrained_weight = self.load_embeddings(vocab_size, emb_size)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=options.drp)

    def forward(self, inp):
        x, x_lengths = inp[0], inp[1]
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        token_emb = self.embed(x)
        token_emb = self.drop(token_emb)
        token_emb = torch.nn.utils.rnn.pack_padded_sequence(token_emb, x_lengths, batch_first=True)
        gru_out, gru_hid = self.rnn(token_emb, h_0)
        # assuming dimension 0, 1 is for layer 1 and 2, 3 for layer 2
        if self.direction == 2:
            gru_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(gru_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                gru_hids.append(x_hid_temp)
            gru_hid = torch.cat(gru_hids, 0)
        # gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # using gru_out and returning gru_out[:, -1, :].unsqueeze(1) is wrong coz its all 0s careful! it doesn't adjust for variable timesteps

        gru_hid = gru_hid[self.num_lyr-1, :, :].unsqueeze(0)
        # take the last layer of the encoder GRU
        gru_hid = gru_hid.transpose(0, 1)

        return gru_hid


    def load_embeddings(self, vocab_size, emb_size):
        vocab_file = './data/vocab.txt'
        embed_file = './data/glove.6B.50d.txt'
        vocab = {}
        embeddings_index = {}
        with open(vocab_file, 'r') as f:
            tokens = [line.strip() for line in f.readlines()]

        for i, token in enumerate(tokens):
            
            vocab[token] = i

        #f = open(embed_file)
        with open(embed_file, 'r') as fp:
            f = fp.readlines()
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) > 301:
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, emb_size))
        for word, i in vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
		#print(embedding_vector)
                embedding_matrix[i] = embedding_vector
        print(embedding_matrix.shape)
        return embedding_matrix

# encode the hidden states of a number of utterances
class InterUtteranceEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, options):
        super(InterUtteranceEncoder, self).__init__()
        self.hid_size = hid_size
        self.rnn = nn.GRU(hidden_size=hid_size, input_size=inp_size,
                          num_layers=1, bidirectional=False, batch_first=True, dropout=options.drp)

    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        # output, h_n for output batch is already dim 0
        h_o, h_n = self.rnn(x, h_0)
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        return h_n



class RetrieveEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, options):
        super(RetrieveEncoder, self).__init__()
        self.use_embed = options.use_embed
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(options.drp)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=9182, sparse=False)
        if self.use_embed:
            pretrained_weight = self.load_embeddings(vocab_size, emb_size)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=options.drp)

    def forward(self, inp):
        x, x_lengths = inp[0], inp[1]
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        token_emb = self.embed(x)
        token_emb = self.drop(token_emb)
        token_emb = torch.nn.utils.rnn.pack_padded_sequence(token_emb, x_lengths, batch_first=True)
        gru_out, gru_hid = self.rnn(token_emb, h_0)
        # assuming dimension 0, 1 is for layer 1 and 2, 3 for layer 2
        if self.direction == 2:
            gru_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(gru_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                gru_hids.append(x_hid_temp)
            gru_hid = torch.cat(gru_hids, 0)
        # gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # using gru_out and returning gru_out[:, -1, :].unsqueeze(1) is wrong coz its all 0s careful! it doesn't adjust for variable timesteps

        gru_hid = gru_hid[self.num_lyr-1, :, :].unsqueeze(0)
        # take the last layer of the encoder GRU
        gru_hid = gru_hid.transpose(0, 1)

        return gru_hid


    def load_embeddings(self, vocab_size, emb_size):
        vocab_file = './data/vocab.txt'
        embed_file = './data/glove.6B.50d.txt'
        vocab = {}
        embeddings_index = {}
        with open(vocab_file, 'r') as f:
            tokens = [line.strip() for line in f.readlines()]

        for i, token in enumerate(tokens):
            
            vocab[token] = i

        #f = open(embed_file)
        with open(embed_file, 'r') as fp:
            f = fp.readlines()
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) > 301:
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, emb_size))
        for word, i in vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
		#print(embedding_vector)
                embedding_matrix[i] = embedding_vector
        print(embedding_matrix.shape)
        return embedding_matrix

class SlotEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, options):
        super(SlotEncoder, self).__init__()
        self.use_embed = options.use_embed
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(options.drp)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=9182, sparse=False)
        if self.use_embed:
            pretrained_weight = self.load_embeddings(vocab_size, emb_size)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=options.drp)

    def forward(self, inp):
        x, x_lengths = inp[0], inp[1]
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        token_emb = self.embed(x)
        token_emb = self.drop(token_emb)
        token_emb = torch.nn.utils.rnn.pack_padded_sequence(token_emb, x_lengths, batch_first=True)
        gru_out, gru_hid = self.rnn(token_emb, h_0)
        # assuming dimension 0, 1 is for layer 1 and 2, 3 for layer 2
        if self.direction == 2:
            gru_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(gru_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                gru_hids.append(x_hid_temp)
            gru_hid = torch.cat(gru_hids, 0)
        # gru_out, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        # using gru_out and returning gru_out[:, -1, :].unsqueeze(1) is wrong coz its all 0s careful! it doesn't adjust for variable timesteps

        gru_hid = gru_hid[self.num_lyr-1, :, :].unsqueeze(0)
        # take the last layer of the encoder GRU
        gru_hid = gru_hid.transpose(0, 1)

        return gru_hid


    def load_embeddings(self, vocab_size, emb_size):
        vocab_file = './data/vocab.txt'
        embed_file = './data/glove.6B.50d.txt'
        vocab = {}
        embeddings_index = {}
        with open(vocab_file, 'r') as f:
            tokens = [line.strip() for line in f.readlines()]

        for i, token in enumerate(tokens):
            
            vocab[token] = i

        #f = open(embed_file)
        with open(embed_file, 'r') as fp:
            f = fp.readlines()
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) > 301:
                continue
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        embedding_matrix = np.zeros((vocab_size, emb_size))
        for word, i in vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
		#print(embedding_vector)
                embedding_matrix[i] = embedding_vector
        print(embedding_matrix.shape)
        return embedding_matrix

# decode the hidden state
class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        self.emb_size = options.emb_size
        self.hid_size = options.dec_hid_size
        self.num_lyr = 1
        self.teacher_forcing = options.teacher
        self.train_lm = options.lm

        self.drop = nn.Dropout(options.drp)
        self.tanh = nn.Tanh()
        self.shared_weight = options.shrd_dec_emb

        self.embed_in = nn.Embedding(options.vocab_size, self.emb_size, padding_idx=9182, sparse=False)
        if not self.shared_weight:
            self.embed_out = nn.Linear(self.emb_size, options.vocab_size, bias=False)

        self.rnn = nn.GRU(hidden_size=self.hid_size,input_size=self.emb_size,num_layers=self.num_lyr,batch_first=True,dropout=options.drp)

        if options.seq2seq:
            self.ses_to_dec = nn.Linear(2 * options.ut_hid_size, self.hid_size)
            self.dec_inf = nn.Linear(self.hid_size, self.emb_size*2, False)
            self.ses_inf = nn.Linear(2 * options.ut_hid_size, self.emb_size*2, False)
        else:
            self.ses_to_dec = nn.Linear(options.ses_hid_size, self.hid_size)
            self.dec_inf = nn.Linear(self.hid_size, self.emb_size*2, False)
            self.ses_inf = nn.Linear(options.ses_hid_size, self.emb_size*2, False)
        self.emb_inf = nn.Linear(self.emb_size, self.emb_size*2, True)
        self.tc_ratio = 1.0

        if options.lm:
            self.lm = nn.GRU(input_size=self.emb_size, hidden_size=self.hid_size, num_layers=self.num_lyr, batch_first=True, dropout=options.drp, bidirectional=False)
            self.lin3 = nn.Linear(self.hid_size, self.emb_size, False)

    def do_decode_tc(self, context_encoding, target, target_lengths):
        #print(target.size(), target_lengths)
        target_emb = self.embed_in(target)
        target_emb = self.drop(target_emb)
        # below will be used later as a crude approximation of an LM
        emb_inf_vec = self.emb_inf(target_emb)

        target_emb = torch.nn.utils.rnn.pack_padded_sequence(target_emb, target_lengths, batch_first=True)

        #print(context_encoding.size())
        init_hidn = self.tanh(self.ses_to_dec(context_encoding))
        #print(init_hidn.size())
        init_hidn = init_hidn.view(self.num_lyr, target.size(0), self.hid_size)

        hid_o, hid_n = self.rnn(target_emb, init_hidn)
        hid_o, _ = torch.nn.utils.rnn.pad_packed_sequence(hid_o, batch_first=True)
        # linear layers not compatible with PackedSequence need to unpack, will be 0s at padded timesteps!

        dec_hid_vec = self.dec_inf(hid_o)
        ses_inf_vec = self.ses_inf(context_encoding)
        #print(dec_hid_vec.size(), ses_inf_vec.size(), emb_inf_vec.size())
        total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec

        hid_o_mx = max_out(total_hid_o)
        hid_o_mx = F.linear(hid_o_mx, self.embed_in.weight) if self.shared_weight else self.embed_out(hid_o_mx)

        if self.train_lm:
            siz = target.size(0)

            lm_hid0 = Variable(torch.zeros(self.num_lyr, siz, self.hid_size))
            if use_cuda:
                lm_hid0 = lm_hid0.cuda()

            lm_o, lm_hid = self.lm(target_emb, lm_hid0)
            lm_o, _ = torch.nn.utils.rnn.pad_packed_sequence(lm_o, batch_first=True)
            lm_o = self.lin3(lm_o)
            lm_o = F.linear(lm_o, self.embed_in.weight) if self.shared_weight else self.embed_out(lm_o)
            return hid_o_mx, lm_o
        else:
            return hid_o_mx, None


    def do_decode(self, siz, seq_len, context_encoding, target):
        ses_inf_vec = self.ses_inf(context_encoding)
        context_encoding = self.tanh(self.ses_to_dec(context_encoding))
        hid_n, preds, lm_preds = context_encoding, [], []

        hid_n = hid_n.view(self.num_lyr, siz, self.hid_size)
        inp_tok = Variable(torch.ones(siz, 1).long())
        lm_hid = Variable(torch.zeros(self.num_lyr, siz, self.hid_size))
        if use_cuda:
            lm_hid = lm_hid.cuda()
            inp_tok = inp_tok.cuda()


        for i in range(seq_len):
            # initially tc_ratio is 1 but then slowly decays to 0 (to match inference time)
            if torch.randn(1)[0] < self.tc_ratio:
                inp_tok = target[:, i].unsqueeze(1)

            inp_tok_embedding = self.embed_in(inp_tok)
            emb_inf_vec = self.emb_inf(inp_tok_embedding)

            inp_tok_embedding = self.drop(inp_tok_embedding)

            hid_o, hid_n = self.rnn(inp_tok_embedding, hid_n)
            dec_hid_vec = self.dec_inf(hid_o)

            total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec
            hid_o_mx = max_out(total_hid_o)

            hid_o_mx = F.linear(hid_o_mx, self.embed_in.weight) if self.shared_weight else self.embed_out(hid_o_mx)
            preds.append(hid_o_mx)

            if self.train_lm:
                lm_o, lm_hid = self.lm(inp_tok_embedding, lm_hid)
                lm_o = self.lin3(lm_o)
                lm_o = F.linear(lm_o, self.embed_in.weight) if self.shared_weight else self.embed_out(lm_o)
                lm_preds.append(lm_o)

            op = hid_o[:, :, :-1]
            op = F.log_softmax(op, 2, 5)
            max_val, inp_tok = torch.max(op, dim=2)
            # now inp_tok will be val between 0 and 10002 ignoring padding_idx
            # here we do greedy decoding
            # so we can ignore the last symbol which is a padding token
            # technically we don't need a softmax here as we just want to choose the max token, max score will result in max softmax.Duh!

        dec_o = torch.cat(preds, 1)
        dec_lmo = torch.cat(lm_preds, 1) if self.train_lm else None
        return dec_o, dec_lmo

    def forward(self, input):
        if len(input) == 1:
            context_encoding = input
            x, x_lengths = None, None
            beam = 5
        elif len(input) == 3:
            context_encoding, x, x_lengths = input
            beam = 5
        else:
            context_encoding, x, x_lengths, beam = input

        if use_cuda:
            x = x.cuda()
        siz, seq_len = x.size(0), x.size(1)

        if self.teacher_forcing:
            dec_o, dec_lm = self.do_decode_tc(context_encoding, x, x_lengths)
        else:
            dec_o, dec_lm = self.do_decode(siz, seq_len, context_encoding, x)

        return dec_o, dec_lm

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val

    def get_teacher_forcing(self):
        return self.teacher_forcing

    def set_tc_ratio(self, new_val):
        self.tc_ratio = new_val

    def get_tc_ratio(self):
        return self.tc_ratio


class HSeq2seq(nn.Module):
    def __init__(self, options):
        super(HSeq2seq, self).__init__()
        self.seq2seq = options.seq2seq
        self.utt_enc = UtteranceEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.intutt_enc = InterUtteranceEncoder(options.ses_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options)

    def forward(self, batch):
        u1, u1_lenghts, u2, u2_lenghts, u3, u3_lenghts = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()

        if self.seq2seq:
            o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))
            qu_seq = torch.cat((o1, o2), 2)
            #final_session_o = self.intutt_enc(qu_seq)
            preds, lmpreds = self.dec((qu_seq, u3, u3_lenghts))
        else:
            o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))
            qu_seq = torch.cat((o1, o2), 1)
            final_session_o = self.intutt_enc(qu_seq)
            preds, lmpreds = self.dec((final_session_o, u3, u3_lenghts))

        return preds, lmpreds


class HSeq2seq_retrieve(nn.Module):
    def __init__(self, options):
        super(HSeq2seq_retrieve, self).__init__()
        self.seq2seq = options.seq2seq
        self.utt_enc = UtteranceEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
       
        self.intutt_enc = InterUtteranceEncoder(options.ses_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options)

    def forward(self, batch):
        u1, u1_lenghts, u2, u2_lenghts, u3, u3_lenghts, ret3, ret3_lenghts = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()

        
        o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))
        o3 = self.utt_enc((u3, u3_lenghts))
        qu_seq = torch.cat((o1, o2, o3), 1)
        final_session_o = self.intutt_enc(qu_seq)
        preds, lmpreds = self.dec((final_session_o, u3, u3_lenghts))

        
        return preds, lmpreds


class HSeq2seq_retrieve_slot(nn.Module):
    def __init__(self, options):
        super(HSeq2seq_retrieve_slot, self).__init__()
        self.seq2seq = options.seq2seq
        self.utt_enc = UtteranceEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.slot_enc = SlotEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.ret_enc = RetrieveEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.intutt_enc = InterUtteranceEncoder(options.ses_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options)

    def forward(self, batch):
        u1, u1_lenghts, u2, u2_lenghts, u3, u3_lenghts, ret3, ret3_lenghts ,slot1, slot1_lenghts, slot2, slot2_lenghts = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7] , batch[8], batch[9], batch[10], batch[11]
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()

        o1, o2 = self.utt_enc((u1, u1_lenghts)), self.utt_enc((u2, u2_lenghts))

        o1_slot, o2_slot = self.slot_enc((slot1, slot1_lenghts)), self.slot_enc((slot2, slot2_lenghts))
        o3 = self.ret_enc((u3, u3_lenghts))
        qu_seq = torch.cat((o1, o1_slots, o2, o2_slots, o3), 1)
        final_session_o = self.intutt_enc(qu_seq)
        preds, lmpreds = self.dec((final_session_o, u3, u3_lenghts))

        
        return preds, lmpreds