import numpy as np

def flvs_check(clients_grads_list):
    def cosine_distance(a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        similiarity = np.dot(a, b.T) / (a_norm * b_norm)
        # dist = 1. - similiarity
        return similiarity

    clients_model_grads_flatten_list = []
    for clients_grads in clients_grads_list:
        temp_client_grads_flatten = np.zeros(0, dtype=np.float64)
        for i in range(len(clients_grads)):
            temp_client_grads_flatten = np.append(temp_client_grads_flatten, clients_grads[i].reshape(-1))
        clients_model_grads_flatten_list.append(temp_client_grads_flatten)

    client_len = len(clients_grads_list)
    yuxianzhi = np.zeros((client_len, client_len), dtype=np.float)
    for i in range(client_len - 1):
        for j in range(i, client_len):
            if i == j:
                continue
            yuxianzhi[i, j] = yuxianzhi[j, i] = cosine_distance(clients_model_grads_flatten_list[i],
                                                                clients_model_grads_flatten_list[j])

    non_malicous_update_index = []
    for i in range(len(clients_model_grads_flatten_list)):
        temp_panduan = np.where(yuxianzhi[i] > 0, 1, 0)
        temp_panduan = np.bincount(temp_panduan)
        if temp_panduan.size > 1 and temp_panduan[1] > temp_panduan[0] - 1:
            non_malicous_update_index.append(i)

    print('selected_clients_id:{}'.format(non_malicous_update_index))
    return non_malicous_update_index

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