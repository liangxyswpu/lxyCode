import numpy as np


def krum(clients_grads_list, malicious_num):
    client_num = len(clients_grads_list)
    non_malicious_count = client_num - malicious_num
    assert client_num > malicious_num

    dis_list = np.zeros((client_num, client_num), dtype=np.float)

    for i in range(client_num):
        for j in range(i):
            s = np.array(clients_grads_list[i], dtype=object) - np.array(clients_grads_list[j], dtype=object)
            ss = []
            for t in s:
                ss.append(t.reshape(-1, ))
            s = np.hstack(np.array(ss, dtype=object))
            dis_list[i][j] = dis_list[j][i] = np.linalg.norm(s) ** 2

    clients_grades = np.zeros((client_num), dtype=np.float)
    for i in range(client_num):
        temp_dis = np.sort(dis_list[i])
        clients_grades[i] = np.sum(temp_dis[:non_malicious_count])

    selected_clients_id = np.argmin(clients_grades)
    print('selected_clients_id:{}'.format(selected_clients_id))
    return selected_clients_id, clients_grads_list[selected_clients_id]


def mean(malicious_clients_num, index, client_num, clients_list):
    grads = np.zeros_like(clients_list[0].grads[index].flatten(), dtype=np.float)

    for j in range(len(grads)):
        temp_grads_list_index_i_layer = []
        for client in clients_list:
            temp_client_grads_index_i_layer = client.grads[index]
            temp_grads_list_index_i_layer.append(temp_client_grads_index_i_layer.flatten()[j])
        temp_grads_list_index_i_layer = np.sort(temp_grads_list_index_i_layer)[
                                        malicious_clients_num:client_num - malicious_clients_num]
        grads[j] = np.mean(temp_grads_list_index_i_layer)
    return grads


def median(malicious_clients_num, index, client_num, clients_list):
    grads = np.zeros_like(clients_list[0].grads[index].flatten(), dtype=np.float)
    for j in range(len(grads)):
        temp_grads_list_index_i_layer = []
        for client in clients_list:
            temp_client_grads_index_i_layer = client.grads[index]
            temp_grads_list_index_i_layer.append(temp_client_grads_index_i_layer.flatten()[j])
        temp_grads_list_index_i_layer = np.sort(temp_grads_list_index_i_layer)[
                                        malicious_clients_num:client_num - malicious_clients_num]
        grads[j] = np.median(temp_grads_list_index_i_layer)
    return grads