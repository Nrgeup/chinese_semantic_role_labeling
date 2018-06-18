#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

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
            flag, name = label[:label.find('-')], label[label.find('-')+1:]
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
                    assert name == lastname, "the I-/E- labels are inconsistent with B- labels in gold file."
                    keys_gold[name][-1] += ' ' + word
        for item in pred:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-')+1:]
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
                    assert name == lastname, "the I-/E- labels are inconsistent with B- labels in pred file."
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
                        keys_gold[key].remove(word) # avoid replicate words
    assert case_recall != 0, "no labels in gold files!"
    assert case_precision != 0, "no labels in pred files!"
    recall = 1.0 * case_true / case_recall
    precision = 1.0 * case_true / case_precision
    f1 = 2.0 * recall * precision / (recall + precision)
    result = "recall: %s  precision: %s  F: %s" % (str(recall), str(precision), str(f1))
    return result
# calc_f1('cpbtest1.txt', 'cpbtest_answer.txt')
if __name__ == "__main__":
    if len(sys.argv[1:]) != 2:
        print('the function takes exactly two parameters: pred_file and gold_file')
    else:
        if not os.path.exists(sys.argv[1]):
            print('pred_file not exists!')
        elif not os.path.exists(sys.argv[2]):
            print('gold_file not exists!')
        else:
            print(calc_f1(sys.argv[1], sys.argv[2]))