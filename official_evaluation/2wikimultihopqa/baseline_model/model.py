"""
    2Wiki-Multihop QA baseline model
    Adapted from HotpotQA Baseline at https://github.com/hotpotqa/hotpot
"""
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn


class Model(nn.Module):
    def __init__(self, config, word_mat, char_mat, relation_mat):
        super().__init__()
        self.config = config
        self.word_dim = config.glove_dim
        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False
        #
        self.relation_dim = config.relation_dim
        self.relation_emb = nn.Embedding(len(relation_mat), len(relation_mat[0]), padding_idx=0)
        self.relation_emb.weight.data.copy_(torch.from_numpy(relation_mat))
        #
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
        self.char_hidden = config.char_hidden
        self.hidden = config.hidden

        self.num_relations = config.num_relations

        self.rnn = EncoderRNN(config.char_hidden+self.word_dim, config.hidden, 1, True, True, 1-config.keep_prob, False)

        self.qc_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_1 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.self_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_2 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_sp = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_sp = nn.Linear(config.hidden*2, 1)

        # Evidence information ------------
        self.rnn_relation = EncoderRNN(config.hidden*2, config.hidden, 1, False, True, 1-config.keep_prob, True)
        self.linear_relation = nn.Linear(config.hidden*2, self.num_relations)
        #
        self.rnn_relation_2 = EncoderRNN(config.relation_dim, config.hidden, 1, True, True, 1-config.keep_prob, False)

        self.rs_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_rs_att = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_relation_3 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)


        self.rnn_subject_start = EncoderRNN(config.hidden*2, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_subject_start = nn.Linear(config.hidden*2, self.num_relations)

        self.rnn_subject_end = EncoderRNN(config.hidden*2, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_subject_end = nn.Linear(config.hidden*2, self.num_relations)

        self.rnn_object_start = EncoderRNN(config.hidden*2, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_object_start = nn.Linear(config.hidden*2, self.num_relations)

        self.rnn_object_end = EncoderRNN(config.hidden*2, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_object_end = nn.Linear(config.hidden*2, self.num_relations)
        # End ------------

        self.rnn_start = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_start = nn.Linear(config.hidden*2, 1)

        self.rnn_end = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_end = nn.Linear(config.hidden*2, 1)

        self.rnn_type = EncoderRNN(config.hidden*3, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.linear_type = nn.Linear(config.hidden*2, 3)

        self.cache_S = 0


    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)


    def handle_outer(self, logit1, logit2):
        outer = logit1.unsqueeze(2) + logit2.unsqueeze(1)
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask.unsqueeze(0))
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return yp1, yp2


    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, relations, context_lens, start_mapping, end_mapping, all_mapping, return_yp=False):
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (context_idxs > 0).float()
        ques_mask = (ques_idxs > 0).float()
        question_lens = (ques_idxs > 0).sum(dim=1)

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

        context_word = self.word_emb(context_idxs)
        ques_word = self.word_emb(ques_idxs)

        context_output = torch.cat([context_word, context_ch], dim=2)
        ques_output = torch.cat([ques_word, ques_ch], dim=2)

        context_output = self.rnn(context_output, context_lens)
        ques_output = self.rnn(ques_output, question_lens)

        # Evidence information ------------
        relation_output = self.rnn_relation(ques_output, question_lens) 
        logit_relation = self.linear_relation(relation_output)

        outputs = nn.Sigmoid()(logit_relation)

        loss_relation = nn.BCELoss(reduction='none')(outputs, relations.float()).sum(dim=1).mean(dim=0)
        k_relations = (outputs >= 0.5) 

        context_relations = self.relation_emb.weight.view(1, self.num_relations, -1) * k_relations.view(-1, self.num_relations, 1).float()
        relations_mask = k_relations.float()
        # End ------------

        output = self.qc_att(context_output, ques_output, ques_mask)
        output = self.linear_1(output)

        output_t = self.rnn_2(output, context_lens)
        output_t = self.self_att(output_t, output_t, context_mask)
        output_t = self.linear_2(output_t)

        output = output + output_t
        # 
        sp_output = self.rnn_sp(output, context_lens)

        start_output = torch.matmul(start_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,self.hidden:])
        end_output = torch.matmul(end_mapping.permute(0, 2, 1).contiguous(), sp_output[:,:,:self.hidden])
        sp_output = torch.cat([start_output, end_output], dim=-1)
        sp_output_t = self.linear_sp(sp_output)
        sp_output_aux = Variable(sp_output_t.data.new(sp_output_t.size(0), sp_output_t.size(1), 1).zero_())
        predict_support = torch.cat([sp_output_aux, sp_output_t], dim=-1).contiguous()

        sp_output = torch.matmul(all_mapping, sp_output)

        # Evidence information ------------
        context_relations = self.rnn_relation_2(context_relations) 

        output_evi = self.rs_att(sp_output, context_relations, relations_mask) 
        output_evi = self.linear_rs_att(output_evi) 

        output_evi = self.rnn_relation_3(output_evi, context_lens)

        # subject
        output_subject_start = self.rnn_subject_start(output_evi, context_lens) 
        logit_subject_start = self.linear_subject_start(output_subject_start) - 1e30 * (1 - context_mask).unsqueeze(2)

        output_subject_end = self.rnn_subject_end(output_subject_start, context_lens) 
        logit_subject_end = self.linear_subject_end(output_subject_end) - 1e30 * (1 - context_mask).unsqueeze(2)

        # object
        output_object_start = self.rnn_object_start(output_subject_end, context_lens)
        logit_object_start = self.linear_object_start(output_object_start) - 1e30 * (1 - context_mask).unsqueeze(2)

        output_object_end = self.rnn_object_end(output_object_start, context_lens) 
        logit_object_end = self.linear_object_end(output_object_end) - 1e30 * (1 - context_mask).unsqueeze(2)
        # End ------------

        # Start token
        output_start = torch.cat([output, output_object_end], dim=-1)

        output_start = self.rnn_start(output_start, context_lens)
        logit1 = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)

        # End token
        output_end = torch.cat([output, output_start], dim=2)

        output_end = self.rnn_end(output_end, context_lens)
        logit2 = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)
 
        output_type = torch.cat([output, output_end], dim=2)
        output_type = torch.max(self.rnn_type(output_type, context_lens), 1)[0]
        predict_type = self.linear_type(output_type)

        if not return_yp: return logit1, logit2, predict_type, predict_support, logit_subject_start, logit_subject_end, logit_object_start, logit_object_end, k_relations, loss_relation

        #
        yp1, yp2 = self.handle_outer(logit1, logit2)
        # Evidence information ------------
        sy1 = [] # sy* is a list of pos of each relation
        sy2 = []
        for i in range(self.num_relations):
            p1, p2 = self.handle_outer(logit_subject_start[:,:,i], logit_subject_end[:,:,i])
            sy1.append(p1)
            sy2.append(p2)
        oy1 = [] # oy* is a list of pos of each relation
        oy2 = []
        for i in range(self.num_relations):
           p1, p2 = self.handle_outer(logit_object_start[:,:,i], logit_object_end[:,:,i])
           oy1.append(p1)
           oy2.append(p2)

        return logit1, logit2, predict_type, predict_support, logit_subject_start, logit_subject_end, logit_object_start, logit_object_end, k_relations, loss_relation, yp1, yp2, sy1, sy2, oy1, oy2


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                # output = rnn.pack_padded_sequence(output, lens, batch_first=True)
                output = rnn.pack_padded_sequence(output, lens, batch_first=True, enforce_sorted=False)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))
