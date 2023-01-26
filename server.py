import multiprocessing
import time

import numpy as np
from sklearn.model_selection import train_test_split
from model import *
from aggregation import *
from utils import *

number_process = 10


class Server():
    def __init__(self, X_train, y_train, model_name, classes_num, malicious_clients_num, *input_shape):
        X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=100)
        self.dataset = {
            'val_x': X_val,
            'val_y': y_val,
            'test_x': X_test,
            'test_y': y_test
        }
        self.malicious_clients_num = malicious_clients_num
        self.model = None
        if model_name == 'CNN':
            self.model = ConvolutionalNetwork()
        elif model_name == 'LR':
            self.model = LogisticRegression(classes_num)
        else:
            self.model = ResNet(classes_num)
        self.model.build(input_shape=input_shape)
        self.model.summary()

    def flvs(self, clients_list):
        clients_model_grads_list = []
        clients_model_grads_flatten_list = []
        for client in clients_list:
            clients_model_grads_list.append(client.grads)

            temp_client_grads_flatten = np.zeros(0, dtype=np.float64)
            for i in range(len(client.grads)):
                temp_client_grads_flatten = np.append(temp_client_grads_flatten, client.grads[i].reshape(-1))
            clients_model_grads_flatten_list.append(temp_client_grads_flatten)

        # cluster
        def cosine_distance(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            similiarity = np.dot(a, b.T) / (a_norm * b_norm)
            # dist = 1. - similiarity
            return similiarity

        client_len = len(clients_list)
        yuxianzhi = np.zeros((client_len, client_len), dtype=np.float)
        for i in range(client_len - 1):
            for j in range(i, client_len):
                if i == j:
                    continue
                yuxianzhi[i, j] = yuxianzhi[j, i] = cosine_distance(clients_model_grads_flatten_list[i],
                                                                    clients_model_grads_flatten_list[j])

        median_updates = np.median(np.array(clients_model_grads_flatten_list), 0)
        median_updates_erfanshu = np.linalg.norm(median_updates)
        non_malicous_update_list = []
        non_malicous_update_index = []
        scores_list = []
        for i in range(len(clients_model_grads_flatten_list)):
            temp_panduan = np.where(yuxianzhi[i] > 0, 1, 0)
            temp_panduan = np.bincount(temp_panduan)
            if temp_panduan.size>1 and temp_panduan[1] > temp_panduan[0] - 1:
                non_malicous_update_index.append(i)
                scores_list.append(np.sum(yuxianzhi[i]))
                temp_bili = median_updates_erfanshu / np.linalg.norm(clients_model_grads_flatten_list[i])
                # temp_bili = min(median_updates_erfanshu / np.linalg.norm(clients_model_grads_flatten_list[i]), 1)
                temp_grad_list = []
                for j in range(len(clients_model_grads_list[i])):
                    temp_grad_list.append(temp_bili * clients_model_grads_list[i][j])
                non_malicous_update_list.append(temp_grad_list)

        scores_sum = np.sum(scores_list)
        # grads_list = []
        # for grads in zip(*non_malicous_update_list):
        #     temp_grads = np.zeros_like(grads[0])
        #     for j in range(len(grads)):
        #         temp_grads += scores_list[j] / scores_sum * grads[j]
        #     grads_list.append(temp_grads)
        print(non_malicous_update_index)

        grads_list = []
        for grads in zip(*non_malicous_update_list):
            grads_list.append(np.mean(grads, axis=0))
        if len(non_malicous_update_list) > 0:
            self.set_weights_by_grads(grads_list, self.model.get_weights())
        # for i in range(len(non_malicous_update_list)):
        #     self.set_weights_by_grads(non_malicous_update_list[i], self.model.get_weights(),
        #                               scores_list[i] / scores_sum)
        # accuracy, poison_accuracy = self.server_model_test(6, 3)
        # print('epoch:{},accuracy:{:.4f},posison_accuracy:{}'.format(i, accuracy, poison_accuracy))
        else:
            print('no update')

    def fltrust(self, clients_list):
        pre_weights_server = self.model.get_weights()
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        self.model.fit(self.dataset['val_x'], self.dataset['val_y'], epochs=1, verbose=0, batch_size=32)
        server_w_difference_value = np.array(self.model.get_weights(), dtype=object) - np.array(pre_weights_server,
                                                                                                dtype=object)
        clients_w = []
        ts_sum = 0.
        server_w_difference_value_vector = matrices_convert_vectors(server_w_difference_value)
        server_update_norm = np.linalg.norm(server_w_difference_value_vector)

        for client in clients_list:
            client_w_difference_value_vector = matrices_convert_vectors(np.array(client.grads))
            ts = relu(cos(np.array(server_w_difference_value_vector), np.array(client_w_difference_value_vector)))
            ts_sum += ts
            client_update_norm = np.linalg.norm(client_w_difference_value_vector)

            temp_grads_list = []
            for grads in client.grads:
                temp_grads_list.append(ts * server_update_norm / client_update_norm * grads)
            clients_w.append(temp_grads_list)

        grads_list = []
        for grads in zip(*clients_w):
            grads_list.append(np.sum(grads, 0) / ts_sum)

        self.set_weights_by_grads(grads_list, pre_weights_server)

    def fedavg(self, clients_list):
        clients_model_grads_list = []
        for client in clients_list:
            clients_model_grads_list.append(client.grads)

        grads_list = []
        for grads in zip(*clients_model_grads_list):
            grads_list.append(np.mean(grads, axis=0))

        self.set_weights_by_grads(grads_list, self.model.get_weights())

    def krum(self, clients_list):
        clients_grads_list = []
        for client in clients_list:
            clients_grads_list.append(client.grads)
        _, grads = krum(clients_grads_list, self.malicious_clients_num)
        self.set_weights_by_grads(grads, self.model.get_weights())

    def trimmed_mean(self, clients_list):
        client_num = len(clients_list)
        assert self.malicious_clients_num < client_num / 2

        grads_list = []
        pool = multiprocessing.Pool(number_process)
        for i in range(len(clients_list[0].grads)):
            grads_list.append(pool.apply_async(mean, (self.malicious_clients_num, i, client_num, clients_list)))

        pool.close()
        pool.join()
        for i in range(len(grads_list)):
            grads_list[i] = grads_list[i].get().reshape(clients_list[0].grads[i].shape)

        self.set_weights_by_grads(grads_list, self.model.get_weights())

    def trimmed_median(self, clients_list):
        client_num = len(clients_list)
        assert self.malicious_clients_num < client_num / 2

        grads_list = []
        pool = multiprocessing.Pool(number_process)
        for i in range(len(clients_list[0].grads)):
            grads_list.append(pool.apply_async(median, (self.malicious_clients_num, i, client_num, clients_list)))

        pool.close()
        pool.join()
        for i in range(len(grads_list)):
            grads_list[i] = grads_list[i].get().reshape(clients_list[0].grads[i].shape)
        self.set_weights_by_grads(grads_list, self.model.get_weights())

    def err(self, clients_list, aggregation_method):
        X_val = self.dataset['val_x']
        y_val = self.dataset['val_y']
        data_num = len(X_val)
        client_num = len(clients_list)
        non_malicious_count = client_num - self.malicious_clients_num
        server_model_weights = self.model.get_weights()

        clients_grads_acc_list = np.zeros((client_num), dtype=np.float)
        for i in range(client_num):
            temp_clients_list = []
            for j in range(client_num):
                if i == j:
                    continue
                temp_clients_list.append(clients_list[j])

            self.model.set_weights(server_model_weights)
            self.aggregation(aggregation_method, temp_clients_list)

            temp_pre = tf.nn.softmax(self.model(X_val), -1)
            ac_count = 0
            for j in range(data_num):
                temp_pre_label = tf.argmax(temp_pre[j], -1)
                if temp_pre_label == y_val[j]:
                    ac_count += 1
            accuracy = ac_count / data_num
            clients_grads_acc_list[i] = accuracy

        client_index = np.argsort(clients_grads_acc_list)[::-1]
        non_malicious_client_index = client_index[:non_malicious_count]
        non_malicious_clients_list = []
        for i in non_malicious_client_index:
            non_malicious_clients_list.append(clients_list[i])

        self.model.set_weights(server_model_weights)
        self.aggregation(aggregation_method, non_malicious_clients_list)

    def lfr(self, clients_list, aggregation_method):
        X_val = self.dataset['val_x']
        y_val = self.dataset['val_y']
        client_num = len(clients_list)
        non_malicious_count = client_num - self.malicious_clients_num
        server_model_weights = self.model.get_weights()
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        clients_grads_loss_list = np.zeros((client_num), dtype=np.float)
        for i in range(client_num):
            temp_clients_list = []
            for j in range(client_num):
                if i == j:
                    continue
                temp_clients_list.append(clients_list[j])

            self.model.set_weights(server_model_weights)
            self.aggregation(aggregation_method, temp_clients_list)

            temp_pre = self.model(X_val)
            loss = scce(y_val, temp_pre)
            clients_grads_loss_list[i] = loss.numpy()

        client_index = np.argsort(clients_grads_loss_list)
        non_malicious_client_index = client_index[:non_malicious_count]
        non_malicious_clients_list = []
        for i in non_malicious_client_index:
            non_malicious_clients_list.append(clients_list[i])

        self.model.set_weights(server_model_weights)
        self.aggregation(aggregation_method, non_malicious_clients_list)

    def union(self, clients_list, aggregation_method):
        X_val = self.dataset['val_x']
        y_val = self.dataset['val_y']
        client_num = len(clients_list)
        non_malicious_count = client_num - self.malicious_clients_num
        data_num = len(X_val)
        server_model_weights = self.model.get_weights()
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        clients_grads_loss_list = np.zeros((client_num), dtype=np.float)
        clients_grads_acc_list = np.zeros((client_num), dtype=np.float)
        for i in range(client_num):
            temp_clients_list = []
            for j in range(client_num):
                if i == j:
                    continue
                temp_clients_list.append(clients_list[j])

            self.model.set_weights(server_model_weights)
            self.aggregation(aggregation_method, temp_clients_list)

            temp_pre = self.model(X_val)
            loss = scce(y_val, temp_pre)
            clients_grads_loss_list[i] = loss.numpy()

            temp_pre = tf.nn.softmax(temp_pre, -1)
            ac_count = 0
            for j in range(data_num):
                temp_pre_label = tf.argmax(temp_pre[j], -1)
                if temp_pre_label == y_val[j]:
                    ac_count += 1
            accuracy = ac_count / data_num
            clients_grads_acc_list[i] = accuracy

        client_index_by_loss = np.argsort(clients_grads_loss_list)
        non_malicious_client_index_by_loss = client_index_by_loss[:non_malicious_count]
        client_index_by_acc = np.argsort(clients_grads_acc_list)[::-1]
        non_malicious_client_index_by_acc = client_index_by_acc[:non_malicious_count]

        client_index = np.intersect1d(non_malicious_client_index_by_acc, non_malicious_client_index_by_loss)
        non_malicious_client_index = client_index[:non_malicious_count]
        non_malicious_clients_list = []
        for i in non_malicious_client_index:
            non_malicious_clients_list.append(clients_list[i])

        self.model.set_weights(server_model_weights)
        self.aggregation(aggregation_method, non_malicious_clients_list)

    def server_model_test(self, classes_num, target_label):
        X_test = self.dataset['test_x']
        y_test = self.dataset['test_y']
        data_num = len(X_test)
        confusion_matrix = np.zeros((classes_num, classes_num), dtype=np.int)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
        for data in zip(test_dataset):
            X, y = data[0]
            temp_pre = tf.nn.softmax(self.model(X), -1)

            for i in range(len(X)):
                temp_pre_label = tf.argmax(temp_pre[i], -1)
                confusion_matrix[int(y[i])][int(temp_pre_label)] += 1

        error_num = np.sum(confusion_matrix) - np.sum(np.array([confusion_matrix[i, i] for i in range(classes_num)]))
        misclassified_as_target_label_num = np.sum(confusion_matrix[:, target_label]) - confusion_matrix[
            target_label, target_label]
        accuracy = (data_num - error_num) / data_num
        posison_accuracy = misclassified_as_target_label_num / (data_num - np.sum(confusion_matrix[target_label, :]))
        return accuracy, posison_accuracy

    def set_weights_by_grads(self, grads_list, server_model_weights):
        new_server_weights = []
        for j in range(len(grads_list)):
            new_server_weights.append(server_model_weights[j] + 1 * grads_list[j])
        self.model.set_weights(new_server_weights)

    def aggregation(self, aggregation_method, clients_list):
        start_time = time.time()
        if aggregation_method == 'fedavg':
            self.fedavg(clients_list)
        elif aggregation_method == 'krum':
            self.krum(clients_list)
        elif aggregation_method == 'mean':
            self.trimmed_mean(clients_list)
        elif aggregation_method == 'median':
            self.trimmed_median(clients_list)
        elif aggregation_method == 'fltrust':
            self.fltrust(clients_list)
        elif aggregation_method == 'flvs':
            self.flvs(clients_list)
        else:
            assert 1 == 0, 'aggregation method error'
        end_time = time.time()
        return end_time-start_time
