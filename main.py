import os
import multiprocessing
from threading import Thread

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from client import *
from server import *
from attack import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.backend.set_floatx('float64')
accuracy_list = []
poison_accuracy_list = []
times_list = []
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parameter = get_parameter(dataset_name='cifar-10', non_iid_p=0.5, malicious_clients_p=0.25,
                              aggregation_method='median', defense_method=None, attack_mothod='ad')

    num_client = parameter['num_client']
    dataset_name = parameter['dataset_name']
    model_name = parameter['model_name']
    input_shape = parameter['input_shape']
    classes_num = parameter['classes_num']
    non_iid_p = parameter['non_iid_p']
    malicious_clients_p = parameter['malicious_clients_p']
    aggregation_method = parameter['aggregation_method']
    defense_method = parameter['defense_method']
    attack_mothod = parameter['attack_mothod']
    global_epoch = 500
    target_label = 3

    X_train, X_test, y_train, y_test = get_datas(dataset_name)
    client_datasets_X, client_datasets_y = make_non_iid_datasets(X_train, y_train, non_iid_p, num_client, classes_num)

    # Initialize client and server
    server = Server(X_test, y_test, model_name, classes_num, int(num_client * malicious_clients_p), *input_shape)
    # weights = np.load('temp_weights.npy', allow_pickle=True)
    # server.model.set_weights(weights)
    # server.server_model_test(classes_num, target_label)

    datasets_index = np.argsort(np.array([len(client_datasets_X[i]) for i in range(len(client_datasets_X))]))
    # datasets_index = np.random.choice(np.arange(num_client), num_client, replace=False)
    benign_clients_list = []
    for i in range(num_client - int(num_client * malicious_clients_p)):
        dataset_index = datasets_index[i]
        benign_clients_list.append(Client(client_datasets_X[dataset_index], client_datasets_y[dataset_index]))
    malicious_clients_list = []
    for i in range(num_client - int(num_client * malicious_clients_p), num_client):
        dataset_index = datasets_index[i]
        malicious_clients_list.append(Client(client_datasets_X[dataset_index], client_datasets_y[dataset_index]))

    # federal training
    for epoch in tqdm(range(global_epoch)):
        server_model_weights = server.model.get_weights()
        pool = multiprocessing.Pool(processes=8)
        begin_result_list = []
        for benign_client in benign_clients_list:
            begin_result_list.append(pool.apply_async(benign_client.client_train, (server_model_weights, model_name, classes_num, *input_shape)))

        pool.close()
        pool.join()

        for i in range(len(begin_result_list)):
            benign_clients_list[i] = begin_result_list[i].get()

        if attack_mothod == 'krum':
            attack_by_krum(malicious_clients_list, server_model_weights, model_name, classes_num, *input_shape)
        elif attack_mothod == 'ad':
            attack_by_krum1(malicious_clients_list, server_model_weights, model_name, classes_num, *input_shape)
        elif attack_mothod == 'scaling':
            attack_by_scaling(malicious_clients_list, classes_num, server_model_weights, model_name, target_label, 6,
                              *input_shape)
        elif attack_mothod == 'label-flipping':
            attack_by_label_flipping(malicious_clients_list, classes_num, server_model_weights, model_name,
                                     *input_shape)
        elif attack_mothod == 'gaussian':
            attack_by_gaussian_noise(malicious_clients_list, server_model_weights, model_name, classes_num,
                                     *input_shape)
        elif attack_mothod == 'mean':
            attack_by_trimmed_mean(malicious_clients_list, server_model_weights, model_name, classes_num,
                                   *input_shape)
        elif attack_mothod == 'median':
            attack_by_trimmed_median(malicious_clients_list, server_model_weights, model_name, classes_num,
                                     *input_shape)
        elif attack_mothod == 'uncertain':
            pool = multiprocessing.Pool(processes=10)
            malicious_result_list = []
            malicious_result_list.append(pool.apply_async(attack_by_uncertain_samples, (malicious_clients_list, server_model_weights, model_name, classes_num,
                                                                                        target_label, *input_shape)))
            pool.close()
            pool.join()
            malicious_result_list = malicious_result_list[0].get()
            for i in range(len(malicious_result_list)):
                malicious_clients_list[i] = malicious_result_list[i]

        if defense_method == None:
            temp_time = server.aggregation(aggregation_method, benign_clients_list + malicious_clients_list)
            times_list.append(temp_time)
        elif defense_method == 'err':
            server.err(benign_clients_list + malicious_clients_list, aggregation_method)
        elif defense_method == 'lfr':
            server.lfr(benign_clients_list + malicious_clients_list, aggregation_method)
        elif defense_method == 'union':
            server.union(benign_clients_list + malicious_clients_list, aggregation_method)
        else:
            assert 1 == 0, 'defense method error'
        accuracy, poison_accuracy = server.server_model_test(classes_num, target_label)
        accuracy_list.append(accuracy)
        poison_accuracy_list.append(poison_accuracy)
        print('epoch:{},accuracy:{:.4f},posison_accuracy:{}'.format(epoch, accuracy, poison_accuracy))

        np.save(
            'nc_{}_d_{}_np_{}_mp_{}_e_{}_ag_{}_d_{}_at_{}.npy'.format(str(num_client), str(dataset_name),
                                                                      str(non_iid_p),
                                                                      str(malicious_clients_p), global_epoch,
                                                                      aggregation_method,

                                                                      defense_method, attack_mothod),
            [accuracy_list, poison_accuracy_list])
        np.save('temp_weights.npy', server.model.get_weights())
        # np.save('minist-flvs-times.npy', times_list)
