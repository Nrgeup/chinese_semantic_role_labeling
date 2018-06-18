# Copyright. All Rights Reserved.
# Author: Wang Ke
# Contact: wangke17[AT]pku.edu.cn
# Discription:
#   role label
#
# =============================

import numpy as np
import tensorflow as tf
import data_helper
import BILSTM_CRF
import time
import os


# Hyper-parameters
def create_hparams():
    timestamp = str(int(time.time()))
    return tf.contrib.training.HParams(
        # file path
        word_dict_file="./data/A_dict.txt",
        pos_dict_file="./data/B_dict.txt",
        role_dict_file="./data/C_dict.txt",
        train_path="./data/train/",
        dev_path="./data/dev/",
        test_path="./data/test/",
        cpbtrain_file="./data/cpbtrain.txt",
        cpbdev_file="./data/cpbdev.txt",
        cpbtest_file="./data/cpbtest.txt",
        a_path='a.txt',
        b_path='b.txt',
        c_path='c.txt',
        a_id_path='a_id.txt',
        b_id_path='b_id.txt',
        c_id_path='c_id.txt',
        timestamp=timestamp,
        save_path="./runs/"+timestamp,

        # data params
        batch_size=128,
        seq_max_len=241,
        word_vocab_size=13000,
        pos_vocab_size=33,
        role_vocab_size=20,
        word2id={},
        pos2id={},
        role2id={},
        id2word={},
        id2pos={},
        id2role={},
        max_f1=0.0,

        # model params
        dropout_rate=0.5,
        hidden_dim=120,
        word_emb_dim=100,
        pos_emb_dim=19,
        learning_rate=0.002,
        num_layers=1,
        # train params
        num_epochs=20000,
        # divice
        gpu=1,
    )


def train():
    # load parameters
    hparams = create_hparams()
    start_time = time.time()
    print("preparing train and dev data")
    # load dict message
    [hparams.word2id, hparams.pos2id, hparams.role2id, hparams.id2word, hparams.id2pos, hparams.id2role] = data_helper.load_dict(hparams=hparams)
    # [word, pos] ==> role
    Train_word, Train_pos, Train_role, Dev_word, Dev_pos, Dev_role = data_helper.get_train(hparams=hparams)
    
    print("building model...")
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu)
        sess = tf.Session(config=config)
        with sess.as_default():
            with tf.device("/gpu:" + str(hparams.gpu)):
                initializer = tf.random_uniform_initializer(-0.1, 0.1)
                with tf.variable_scope("model", reuse=None, initializer=initializer):
                    model = BILSTM_CRF.bilstm_crf(hparams=hparams)
                print("training model...")
                sess.run(tf.global_variables_initializer())
                model.train(sess, hparams, Train_word, Train_pos, Train_role, Dev_word, Dev_pos, Dev_role)
                print("final best f1 on valid dataset is: %f" % hparams.max_f1)

    end_time = time.time()
    print("time used %f (hour)" % ((end_time - start_time) / 3600))
    return


def eval():
    hparams = create_hparams()
    # load dict message
    [hparams.word2id, hparams.pos2id, hparams.role2id, hparams.id2word, hparams.id2pos,
     hparams.id2role] = data_helper.load_dict(hparams=hparams)
    print("Evaluation model...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu)
    name = "1513764280"
    checkpoint_dir = os.path.join('runs', name, "checkpoints")
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        with sess.as_default():
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = BILSTM_CRF.bilstm_crf(hparams=hparams)
            # model.saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            sess.run(tf.tables_initializer())
            print("Load ckpt %s file success!" % checkpoint_file)
            type = 'test'
            Test_word, Test_pos, Test_role = data_helper.get_test(hparams, type)
            model.eval(sess, hparams, Test_word, Test_pos, Test_role, type, name)
    return


if __name__ == '__main__':
    train()
    eval()

