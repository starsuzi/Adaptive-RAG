"""
    2Wiki-Multihop QA baseline model
    Adapted from HotpotQA Baseline at https://github.com/hotpotqa/hotpot
"""
import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from model import Model
from util import convert_tokens, evaluate
from util import get_buckets, DataIterator, IGNORE_INDEX
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    #
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduce=False, ignore_index=IGNORE_INDEX)


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32) 
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.relation_emb_file, "r") as fh:
        relation_mat = np.array(json.load(fh), dtype=np.float32) 
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh) 
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh) 

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=['run.py', 'model.py', 'util.py'])
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file)
    dev_buckets = get_buckets(config.dev_record_file)

    def build_train_iterator():
        return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, \
            config.char_limit, True, config.sent_limit, config.num_relations)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, \
            config.char_limit, False, config.sent_limit, config.num_relations)

    model = Model(config, word_mat, char_mat, relation_mat)

    logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    ori_model = model.cuda()
    model = nn.DataParallel(ori_model)

    lr = config.init_lr
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    cur_patience = 0
    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    model.train()

    for epoch in range(10000):
        for data in build_train_iterator():
            context_idxs = Variable(data['context_idxs'])
            ques_idxs = Variable(data['ques_idxs'])
            context_char_idxs = Variable(data['context_char_idxs'])
            ques_char_idxs = Variable(data['ques_char_idxs'])
            #
            context_lens = Variable(data['context_lens'])
            y1 = Variable(data['y1'])
            y2 = Variable(data['y2'])
            q_type = Variable(data['q_type'])
            is_support = Variable(data['is_support'])
            start_mapping = Variable(data['start_mapping'])
            end_mapping = Variable(data['end_mapping'])
            all_mapping = Variable(data['all_mapping'])
            #
            subject_y1 = Variable(data['subject_y1']) 
            subject_y2 = Variable(data['subject_y2'])
            object_y1 = Variable(data['object_y1']) 
            object_y2 = Variable(data['object_y2'])
            relations = Variable(data['relations'])
            # 
            #
            model_results = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, relations, \
                context_lens, start_mapping, end_mapping, all_mapping, return_yp=False)
            # 
            (logit1, logit2, predict_type, predict_support, logit_subject_start, logit_subject_end, \
                logit_object_start, logit_object_end, k_relations, loss_relation) = model_results
            loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
            loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
            loss_3_r = torch.sum(loss_relation)
            loss_3_s = (nll_sum(logit_subject_start, subject_y1) + nll_sum(logit_subject_end, subject_y2)) / context_idxs.size(0)
            loss_3_o = (nll_sum(logit_object_start, object_y1) + nll_sum(logit_object_end, object_y2)) / context_idxs.size(0)
            # 
            loss = loss_1 + config.sp_lambda * loss_2 + config.evi_lambda * (loss_3_s + loss_3_r + loss_3_o)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() # total_loss += loss.data[0]
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(epoch, global_step, lr, elapsed*1000/config.period, cur_loss))
                total_loss = 0
                start_time = time.time()

            if global_step % config.checkpoint == 0:
                model.eval()
                metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file, config)
                model.train()

                logging('-' * 89)
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(global_step//config.checkpoint,
                    epoch, time.time()-eval_start_time, metrics['loss'], metrics['exact_match'], metrics['f1']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0
        if stop_train: break
    logging('best_dev_F1 {}'.format(best_dev_F1))


def evaluate_batch(data_source, model, max_batches, eval_file, config):
    answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
    for step, data in enumerate(iter):
        if step >= max_batches and max_batches > 0: break

        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        y1 = Variable(data['y1'], volatile=True)
        y2 = Variable(data['y2'], volatile=True)
        q_type = Variable(data['q_type'], volatile=True)
        is_support = Variable(data['is_support'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)
        #
        subject_y1 = Variable(data['subject_y1']) 
        subject_y2 = Variable(data['subject_y2'])
        object_y1 = Variable(data['object_y1']) 
        object_y2 = Variable(data['object_y2'])
        relations = Variable(data['relations'])
        # 
        # 
        model_results = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, relations, \
            context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)

        (logit1, logit2, predict_type, predict_support, logit_subject_start, logit_subject_end, \
            logit_object_start, logit_object_end, k_relations, loss_relation, yp1, yp2, sy1, sy2, oy1, oy2) = model_results
        loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
        loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
        loss_3_r = torch.sum(loss_relation)
        loss_3_s = (nll_sum(logit_subject_start, subject_y1) + nll_sum(logit_subject_end, subject_y2)) / context_idxs.size(0)
        loss_3_o = (nll_sum(logit_object_start, object_y1) + nll_sum(logit_object_end, object_y2)) / context_idxs.size(0)

        loss = loss_1 + config.sp_lambda * loss_2 + config.evi_lambda * (loss_3_s + loss_3_r + loss_3_o)

        
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        total_loss += loss.item() # total_loss += loss.data[0]
        step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(eval_file, answer_dict)
    metrics['loss'] = loss

    return metrics


def predict(data_source, model, eval_file, config, prediction_file, idx2relation_dict):
    answer_dict = {}
    sp_dict = {}
    evidence_dict = {}
    sp_th = config.sp_threshold
    for step, data in enumerate(tqdm(data_source)):
        context_idxs = Variable(data['context_idxs'], volatile=True)
        ques_idxs = Variable(data['ques_idxs'], volatile=True)
        context_char_idxs = Variable(data['context_char_idxs'], volatile=True)
        ques_char_idxs = Variable(data['ques_char_idxs'], volatile=True)
        context_lens = Variable(data['context_lens'], volatile=True)
        start_mapping = Variable(data['start_mapping'], volatile=True)
        end_mapping = Variable(data['end_mapping'], volatile=True)
        all_mapping = Variable(data['all_mapping'], volatile=True)
        #
        subject_y1 = Variable(data['subject_y1']) # Size([24, 8])
        subject_y2 = Variable(data['subject_y2'])
        object_y1 = Variable(data['object_y1']) 
        object_y2 = Variable(data['object_y2'])
        relations = Variable(data['relations'])
        # 
        model_results = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, relations, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        
        (logit1, logit2, predict_type, predict_support, logit_subject_start, logit_subject_end, logit_object_start, logit_object_end, k_relations, loss_relation, yp1, yp2, sy1, sy2, oy1, oy2) = model_results
        
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)
        # Sentence-level SPs
        predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy() # (24, 81)
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = data['ids'][i]
            for j in range(predict_support_np.shape[1]):
                if j >= len(eval_file[cur_id]['sent2title_ids']): break
                if predict_support_np[i, j] > sp_th:
                    cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
            sp_dict.update({cur_id: cur_sp_pred})
        # Evidence
        sy1 = [_.data.cpu().numpy() for _ in sy1] # int matrix 
        sy2 = [_.data.cpu().numpy() for _ in sy2] # int matrix 
        oy1 = [_.data.cpu().numpy() for _ in oy1] # int matrix 
        oy2 = [_.data.cpu().numpy() for _ in oy2] # int matrix 

        for i in range(k_relations.shape[0]):
            cur_evi_pred = []
            cur_id = data['ids'][i]
            tuples = []
            for j in range(k_relations.shape[1]):
                if k_relations[i, j] == 1:
                    relation = idx2relation_dict[str(j)]
                    # get subject
                    context = eval_file[str(cur_id)]["context"]
                    spans = eval_file[str(cur_id)]["spans"]
                    # 
                    s_start = sy1[j][i]
                    s_end = sy2[j][i]
                    #
                    start_idx = spans[s_start][0]
                    end_idx = spans[s_end][1]
                    subject = context[start_idx: end_idx]
                    # get object
                    object_start_idx = spans[oy1[j][i]][0]
                    object_end_idx = spans[oy2[j][i]][1]
                    # 
                    object_ = context[object_start_idx: object_end_idx]
                    tuples.append((subject, relation, object_))
            evidence_dict.update({cur_id: tuples})

    prediction = {'answer': answer_dict, 'sp': sp_dict, 'evidence': evidence_dict}
    with open(prediction_file, 'w') as f:
        json.dump(prediction, f)


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    if config.data_split == 'dev':
        with open(config.dev_eval_file, "r") as fh:
            dev_eval_file = json.load(fh)
    else:
        with open(config.test_eval_file, 'r') as fh:
            dev_eval_file = json.load(fh)
    with open(config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)
    with open(config.relation_emb_file, "r") as fh:
        relation_mat = np.array(json.load(fh), dtype=np.float32) # (20, 100)
    with open(config.idx2relation_file, 'r') as fh:
        idx2relation_dict = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    if config.data_split == 'dev':
        dev_buckets = get_buckets(config.dev_record_file)
        para_limit = config.para_limit
        ques_limit = config.ques_limit
    elif config.data_split == 'test':
        para_limit = None
        ques_limit = None
        dev_buckets = get_buckets(config.test_record_file)

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, para_limit,
            ques_limit, config.char_limit, False, config.sent_limit,
            config.num_relations)

    model = Model(config, word_mat, char_mat, relation_mat)

    ori_model = model.cuda()
    ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
    model = nn.DataParallel(ori_model)

    model.eval()
    predict(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file, idx2relation_dict)

