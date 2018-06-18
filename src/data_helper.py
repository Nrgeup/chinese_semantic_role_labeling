import re
import os
import csv
import pandas as pd
import numpy as np


def build_dict(dict_file):
    line_id = 0
    token2id = {}
    id2token = {}
    with open(dict_file) as infile:
        for row in infile:
            token = row.strip()
            if token == "":
                break
            token_id = line_id
            token2id[token] = token_id
            id2token[token_id] = token
            line_id += 1
    return token2id, id2token


def load_dict(hparams):
    # word dict
    word2id, id2word = build_dict(hparams.word_dict_file)
    pos2id, id2pos = build_dict(hparams.pos_dict_file)
    role2id, id2role = build_dict(hparams.role_dict_file)
    return [word2id, pos2id, role2id, id2word, id2pos, id2role]


def get_test(hparams, type):
    Test_word = []
    Test_pos = []
    Test_role = []
    if type == 'test':
        path = hparams.test_path
    if type == 'dev':
        path = hparams.dev_path
    with open(path + hparams.a_id_path) as f_in:  # word
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Test_word.append(uu)
    with open(path + hparams.b_id_path) as f_in:  # pos
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Test_pos.append(uu)
    with open(path + hparams.c_id_path) as f_in:  # role
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Test_role.append(uu)
    assert len(Test_word) == len(Test_pos)
    assert len(Test_pos) == len(Test_role)
    print("Load %s size: %d" % (type, len(Test_word)))
    Test_word = np.array(padding(Test_word, seq_max_len=hparams.seq_max_len))
    Test_pos = np.array(padding(Test_pos, seq_max_len=hparams.seq_max_len))
    Test_role = np.array(padding(Test_role, seq_max_len=hparams.seq_max_len))
    return Test_word, Test_pos, Test_role


def get_train(hparams):
    # load train file
    Train_word = []
    Train_pos = []
    Train_role = []
    path = hparams.train_path
    with open(path + hparams.a_id_path) as f_in:  # word
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Train_word.append(uu)
    with open(path + hparams.b_id_path) as f_in:  # pos
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Train_pos.append(uu)
    with open(path + hparams.c_id_path) as f_in:  # role
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Train_role.append(uu)

    # load dev file
    Dev_word = []
    Dev_pos = []
    Dev_role = []
    path = hparams.dev_path
    with open(path + hparams.a_id_path) as f_in:  # word
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Dev_word.append(uu)
    with open(path + hparams.b_id_path) as f_in:  # pos
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Dev_pos.append(uu)
    with open(path + hparams.c_id_path) as f_in:  # role
        for row_item in f_in:
            _list = row_item.strip().split(' ')
            uu = [int(tmp) for tmp in _list if tmp != '']
            if len(uu) == 0:
                continue
            Dev_role.append(uu)

    print("train size: %d, validation size: %d" % (len(Train_word), len(Dev_word)))

    # padding
    Train_word = np.array(padding(Train_word, seq_max_len=hparams.seq_max_len))
    Train_pos = np.array(padding(Train_pos, seq_max_len=hparams.seq_max_len))
    Train_role = np.array(padding(Train_role, seq_max_len=hparams.seq_max_len))

    Dev_word = np.array(padding(Dev_word, seq_max_len=hparams.seq_max_len))
    Dev_pos = np.array(padding(Dev_pos, seq_max_len=hparams.seq_max_len))
    Dev_role = np.array(padding(Dev_role, seq_max_len=hparams.seq_max_len))

    return Train_word, Train_pos, Train_role, Dev_word, Dev_pos, Dev_role


def padding(sample, seq_max_len):
    """use '0' to padding the sentence"""
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample


def unpadding(sample):
    """delete '0' from padding sentence"""
    sample_new = []
    for item in sample:
        _list = []
        _list_tmp = []
        for ii in item:
            _list_tmp.append(ii)
            if ii != 0:
                _list = _list + _list_tmp
                _list_tmp = []
        sample_new.append(_list)
    return sample_new


def next_test_batch(X_word, X_pos, y_role, start_index, batch_size=128):
    full_size = batch_size
    last_index = start_index + batch_size
    X_word_batch = list(X_word[start_index:min(last_index, len(X_word))])
    X_pos_batch = list(X_pos[start_index:min(last_index, len(X_pos))])
    y_role_batch = list(y_role[start_index:min(last_index, len(y_role))])
    if last_index > len(X_word):
        full_size = len(X_word) - start_index
        left_size = last_index - (len(X_word))
        for i in range(left_size):
            index = np.random.randint(len(X_word))
            X_word_batch.append(X_word[index])
            X_pos_batch.append(X_pos[index])
            y_role_batch.append(y_role[index])
    X_word_batch = np.array(X_word_batch)
    X_pos_batch = np.array(X_pos_batch)
    y_role_batch = np.array(y_role_batch)
    return X_word_batch, X_pos_batch, y_role_batch, full_size



def next_batch(X_word, X_pos, y_role, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_word_batch = list(X_word[start_index:min(last_index, len(X_word))])
    X_pos_batch = list(X_pos[start_index:min(last_index, len(X_pos))])
    y_role_batch = list(y_role[start_index:min(last_index, len(y_role))])
    if last_index > len(X_word):
        left_size = last_index - (len(X_word))
        for i in range(left_size):
            index = np.random.randint(len(X_word))
            X_word_batch.append(X_word[index])
            X_pos_batch.append(X_pos[index])
            y_role_batch.append(y_role[index])
    X_word_batch = np.array(X_word_batch)
    X_pos_batch = np.array(X_pos_batch)
    y_role_batch = np.array(y_role_batch)
    return X_word_batch, X_pos_batch, y_role_batch


def extract_entity(seqs, labels):
    entitys = {}
    for id, item in enumerate(labels):
        if item == 'O' or item == '_PAD' or item == 'rel':
            continue
        if item in entitys:
            entitys[item].append(seqs[id])
        else:
            entitys[item] = [seqs[id]]
    return entitys


def get_length_by_vec(seq_x):
    seq_len = []
    for ii in seq_x:
        _len = len([jj for jj in ii if jj != 0])
        seq_len.append(_len)
        # print(ii)
        assert _len != 0
    seq_len = np.array(seq_len)
    assert len(seq_len) == len(seq_x)
    return seq_len


def next_random_batch(Dev_word, Dev_pos, Dev_role, batch_size):
    x_word_batch = []
    x_pos_batch = []
    y_role_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(Dev_word))
        if len(Dev_word[index]) == 0:
            continue
        x_word_batch.append(Dev_word[index])
        x_pos_batch.append(Dev_pos[index])
        y_role_batch.append(Dev_role[index])
    x_word_batch = np.array(x_word_batch)
    x_pos_batch = np.array(x_pos_batch)
    y_role_batch = np.array(y_role_batch)
    return x_word_batch, x_pos_batch, y_role_batch


log_mode = 'w'


def log(str, out_path):
    global log_mode
    log_path = out_path + "/log.txt"
    with open(log_path, log_mode) as f:
        f.write(str + '\n')
    if log_mode != 'a':
        log_mode = 'a'
    return


def recover_role(role_list):
    ans_list = role_list
    good_list = ['O', 'rel', '_PAD']
    last_item = None
    for i in range(len(role_list)):
        item = ans_list[i]
        next_item = None
        if i != len(role_list) - 1:
            next_item = ans_list[i+1]
        if item == '_PAD':
            print("Error, echo _PAD, %s" % str(role_list))
        if item not in good_list:
            if item != last_item and item != next_item:
                ans_list[i] = 'S-' + ans_list[i]
            if item != last_item and item == next_item:
                ans_list[i] = 'B-' + ans_list[i]
            if item == last_item and item == next_item:
                ans_list[i] = 'I-' + ans_list[i]
            if item == last_item and item != next_item:
                ans_list[i] = 'E-' + ans_list[i]
        last_item = item
    return ans_list


def recover_eval(test_file, outputs_role):
    with open(test_file, 'r') as f:
        outputs_lines = f.readlines()

    # assert len(outputs_lines) == len(outputs_role)
    outputs = []
    for i in range(len(outputs_lines)):
        item_lists = []
        _line = outputs_lines[i].strip()
        if _line == "":
            break
        _line_items = _line.split(' ')

        for j in range(len(_line_items)):
            _item = _line_items[j]
            
            _item_list = _item.split('/')
            _a = _item_list[0]
            _b = _item_list[1]
            _c = outputs_role[i][j]
            if len(_item_list) == 3 and _item_list[2] == 'rel':
                _c = _item_list[2]
            item_lists.append('/'.join([_a, _b, _c]))
        outputs.append(' '.join(item_lists) + '\n')
    return outputs


def calc_f1(pred_file, gold_file):
    case_true, case_recall, case_precision = 0, 0, 0
    golds = [gold.split() for gold in open(gold_file, 'r').read().strip().split('\n')]
    preds = [pred.split() for pred in open(pred_file, 'r').read().strip().split('\n')]
    assert len(golds) == len(preds), "length of prediction file and gold file should be the same."
    for gold, pred in zip(golds, preds):
        lastname = ''
        keys_gold, keys_pred = {}, {}
        for item in gold:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-') + 1:]
            if flag == 'O':
                continue
            if flag == 'S':
                if name not in keys_gold:
                    keys_gold[name] = [word]
                else:
                    keys_gold[name].append(word)
            else:
                if flag == 'B':
                    if name not in keys_gold:
                        keys_gold[name] = [word]
                    else:
                        keys_gold[name].append(word)
                    lastname = name
                elif flag == 'I' or flag == 'E':
                    # assert name == lastname, "the I-/E- labels are inconsistent with B- labels in gold file. %s" % str(gold)
                    keys_gold[name][-1] += ' ' + word
        for item in pred:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-') + 1:]
            if flag == 'O':
                continue
            if flag == 'S':
                if name not in keys_pred:
                    keys_pred[name] = [word]
                else:
                    keys_pred[name].append(word)
            else:
                if flag == 'B':
                    if name not in keys_pred:
                        keys_pred[name] = [word]
                    else:
                        keys_pred[name].append(word)
                    lastname = name
                elif flag == 'I' or flag == 'E':
                    # assert name == lastname, "the I-/E- labels are inconsistent with B- labels in pred file. %s" % str(pred)
                    keys_pred[name][-1] += ' ' + word

        for key in keys_gold:
            case_recall += len(keys_gold[key])
        for key in keys_pred:
            case_precision += len(keys_pred[key])

        for key in keys_pred:
            if key in keys_gold:
                for word in keys_pred[key]:
                    if word in keys_gold[key]:
                        case_true += 1
                        keys_gold[key].remove(word)  # avoid replicate words
    assert case_recall != 0, "no labels in gold files!"
    assert case_precision != 0, "no labels in pred files!"
    recall = 1.0 * case_true / case_recall
    precision = 1.0 * case_true / case_precision
    f1 = 2.0 * recall * precision / (recall + precision)
    return recall, precision, f1

