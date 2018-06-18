import math
import data_helper
import numpy as np
import tensorflow as tf
import time
import shutil
import os


class bilstm_crf(object):

    def __init__(self, hparams, is_training=True):
        # Parameter
        self.num_layers = hparams.num_layers
        self.learning_rate = hparams.learning_rate
        self.hidden_dim = hparams.hidden_dim  # hidden_dim=100
        self.word_emb_dim = hparams.word_emb_dim  # word_emb_dim=90
        self.pos_emb_dim = hparams.pos_emb_dim  # pos_emb_dim=10
        self.dropout_rate = hparams.dropout_rate  # 0.5
        self.word_vocab_size = hparams.word_vocab_size  # 10000,
        self.pos_vocab_size = hparams.pos_vocab_size  # 33,
        self.num_classes = hparams.role_vocab_size  # 20,

        # placeholder of word\pos\role
        self.inputs_word = tf.placeholder(tf.int32, shape=[None, None], name="inputs_word")
        self.inputs_pos = tf.placeholder(tf.int32, shape=[None, None], name="inputs_pos")
        self.predicts_role = tf.placeholder(tf.int32, shape=[None, None], name="predicts_role")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.rel_vector = tf.placeholder(tf.float32, shape=[None, None, 1], name="rel_vector")

        with tf.variable_scope("input-embedding"):
            self.word_embedding = tf.get_variable("emb-word", [self.word_vocab_size, self.word_emb_dim])
            self.pos_embedding = tf.get_variable("emb-pos", [self.word_vocab_size, self.pos_emb_dim])
            self.inputs_emb_word = tf.nn.embedding_lookup(self.word_embedding, self.inputs_word)
            self.inputs_emb_pos = tf.nn.embedding_lookup(self.pos_embedding, self.inputs_pos)

        with tf.variable_scope("concat"):
            self.inputs_emb = tf.concat([self.inputs_emb_word, self.inputs_emb_pos, self.rel_vector], axis=2)

        with tf.variable_scope("bi-lstm"):
            # lstm cell
            lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

            # dropout
            if is_training:
                lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
                lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

            # forward and backward
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell_fw,
                cell_bw=lstm_cell_bw,
                inputs=self.inputs_emb,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32,
            )
            self.output = tf.concat([output_fw, output_bw], axis=-1)

        # project
        with tf.variable_scope("project"):
            W = tf.get_variable("W", shape=[self.hidden_dim * 2, self.num_classes],
                                dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.num_classes], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            nsteps = tf.shape(self.output)[1]
            output = tf.reshape(self.output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.num_classes])

        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        self.logits, self.predicts_role, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        # for tensorboard
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, hparams, Train_word, Train_pos, Train_role, Dev_word, Dev_pos, Dev_role):
        Test_word, Test_pos, Test_role = data_helper.get_test(hparams, type='test')
        checkpoint_dir = hparams.save_path + "/checkpoints"
        checkpoint_prefix = checkpoint_dir + "/model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        merged = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter(hparams.save_path + '/train_loss', sess.graph)

        num_iterations = int(math.ceil(1.0 * len(Train_word) / hparams.batch_size))

        cnt = 0
        for epoch in range(hparams.num_epochs):
            print("current epoch: %d" % (epoch))

            for iteration in range(num_iterations):
                # train
                X_word_train_batch, X_pos_train_batch, y_role_train_batch = data_helper.next_batch(Train_word, Train_pos, Train_role,
                                                                start_index=iteration * hparams.batch_size,
                                                                batch_size=hparams.batch_size)
                X_rel_train_batch = self.get_one_hot_rel(y_role_train_batch, hparams.role2id['rel'])

                X_train_sequence_lengths = data_helper.get_length_by_vec(X_word_train_batch)
                _, loss_train, logits, train_summary = \
                    sess.run([
                        self.optimizer,
                        self.loss,
                        self.logits,
                        self.train_summary
                    ],
                        feed_dict={
                            self.inputs_word: X_word_train_batch,
                            self.inputs_pos: X_pos_train_batch,
                            self.rel_vector: X_rel_train_batch,
                            self.sequence_lengths: X_train_sequence_lengths,
                            self.predicts_role: y_role_train_batch,
                    })

                if iteration % 10 == 0:
                    cnt += 1
                    feed_dict = {
                        self.inputs_word: X_word_train_batch,
                        self.inputs_pos: X_pos_train_batch,
                        self.rel_vector: X_rel_train_batch,
                        self.sequence_lengths: X_train_sequence_lengths,
                        # self.predicts_role: y_role_train_batch,
                    }
                    predicts_train = self.predict(sess, feed_dict, X_train_sequence_lengths)
                    precision_train, recall_train, f1_train = self.evaluate(X_train_sequence_lengths, X_word_train_batch, X_pos_train_batch, y_role_train_batch, predicts_train, hparams.id2word, hparams.id2pos, hparams.id2role)
                    summary_writer_train.add_summary(train_summary, cnt)
                    print("iteration: %3d, train loss: %5f, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (iteration, loss_train, precision_train, recall_train, f1_train))

                # validation
                if iteration % 100 == 0 and f1_train > 0.6:
                    self.eval(sess, hparams, Dev_word, Dev_pos, Dev_role, eval_type='dev', name=hparams.timestamp)
                    precision_dev, recall_dev, f1_dev = data_helper.calc_f1(hparams.cpbdev_file, hparams.save_path + "/eval_dev.txt")
                    print(
                        "iteration: %3d, valid precision: %.5f, valid recall: %.5f, valid f1: %.5f" % (
                        iteration, precision_dev, recall_dev, f1_dev))

                    if f1_dev >= hparams.max_f1:
                        hparams.max_f1 = f1_dev
                        save_name = self.saver.save(sess, checkpoint_prefix, global_step=cnt)
                        shutil.copyfile(hparams.save_path + "/eval_dev.txt", hparams.save_path + "/best_eval_dev.txt")

                        self.eval(sess, hparams, Test_word, Test_pos, Test_role, eval_type='test', name=hparams.timestamp)

                        str_out = "saved the best model with f1: %.5f save path:%s" % (hparams.max_f1, save_name)
                        print(str_out)
                        data_helper.log(str_out, hparams.save_path)

    def predict(self, sess, fd, sequence_lengths):
        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = sess.run(
            [self.logits, self.trans_params], feed_dict=fd
        )

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]
        return viterbi_sequences

    def evaluate(self, lengths, X_word, X_pos, y_true, y_pred, id2word, id2pos, id2role):

        case_true, case_recall, case_precision = 0, 0, 0

        x_word_id = data_helper.unpadding(X_word)
        x_pos_id = data_helper.unpadding(X_pos)
        y_true_id = data_helper.unpadding(y_true)
        y_pred_id = y_pred

        for i in range(len(lengths)):

            x_word = [id2word[val] for val in x_word_id[i]]
            x_pos = [id2pos[val] for val in x_pos_id[i]]
            y = [id2role[val] for val in y_true_id[i]]
            y_hat = [id2role[val] for val in y_pred_id[i]]

            true_labels = data_helper.extract_entity(x_word, y)
            pred_labels = data_helper.extract_entity(x_word, y_hat)

            for key in true_labels:
                case_recall += len(true_labels[key])
            for key in pred_labels:
                case_precision += len(pred_labels[key])

            for key in pred_labels:
                if key in true_labels:
                    for word in pred_labels[key]:
                        if word in true_labels[key]:
                            case_true += 1
                            true_labels[key].remove(word)  # avoid replicate words
        recall = -1.0
        precision = -1.0
        f1 = -1.0
        if case_recall != 0:
            recall = 1.0 * case_true / case_recall
        if case_precision != 0:
            precision = 1.0 * case_true / case_precision
        if recall > 0 and precision > 0:
            f1 = 2.0 * recall * precision / (recall + precision)
        return precision, recall, f1

    def reconstruct(self, lens, roles, id2role):
        ans_seq = []
        for i in range(lens):
            role_list = [id2role[val] for val in roles[i]]
            role_list = data_helper.recover_role(role_list)
            ans_seq.append(role_list)
        return ans_seq

    def get_one_hot_rel(self, vec, ref_id):
        ans = np.zeros(shape=[len(vec), len(vec[0]), 1], dtype=float)
        for i in range(len(vec)):
            j = np.where(vec[i] == ref_id)
            ans[i][j] = [1.0]
        return ans

    def eval(self, sess, hparams, Test_word, Test_pos, Test_role, eval_type, name):
        num_iterations = int(math.ceil(1.0 * len(Test_word) / hparams.batch_size))
        outputs_role = []
        for iteration in range(num_iterations):
            X_word_test_batch, X_pos_test_batch, y_role_test_batch, full_size = data_helper.next_test_batch(Test_word, Test_pos,
                                                                                               Test_role,
                                                                                               start_index=iteration * hparams.batch_size,
                                                                                               batch_size=hparams.batch_size)
            X_rel_test_batch = self.get_one_hot_rel(y_role_test_batch, hparams.role2id['rel'])
            X_test_sequence_lengths = data_helper.get_length_by_vec(X_word_test_batch)

            feed_dict = {
                self.inputs_word: X_word_test_batch,
                self.inputs_pos: X_pos_test_batch,
                self.rel_vector: X_rel_test_batch,
                self.sequence_lengths: X_test_sequence_lengths,
            }
            predicts_dev = self.predict(sess, feed_dict, X_test_sequence_lengths)

            outputs_role += self.reconstruct(full_size, predicts_dev, hparams.id2role)

        if eval_type == 'dev':
            eval_file = hparams.cpbdev_file
        if eval_type == 'test':
            eval_file = hparams.cpbtest_file

        outputs = data_helper.recover_eval(eval_file, outputs_role)

        save_path = "./runs/%s/eval_%s.txt" % (name, eval_type)
        with open(save_path, 'w') as f:
            f.writelines(outputs)
            f.write('\n') # for consistance with cpttest.txt
        print("eval success!, size: %d save at %s" % (len(outputs), save_path))
        return

