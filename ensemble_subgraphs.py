import argparse
import torch
import random
import numpy as np
from sklearn.svm import SVC, LinearSVC
from util import print_time_info, set_random_seed, get_hits, topk
from tqdm import tqdm

def sim_standardization(sim):
    mean = np.mean(sim)
    std = np.std(sim)
    sim = (sim - mean) / std
    return sim

def load_partial_sim(sim_path, standardization=True):
    partial_sim = np.load(sim_path)
    sim_matrix = partial_sim[0]
    sim_indice = partial_sim[1]
    sim_indice = sim_indice.astype(np.int)
    assert sim_matrix.shape == sim_indice.shape
    if standardization:
        sim_matrix = sim_standardization(sim_matrix)
    size = sim_matrix.shape[0]
    sim = np.zeros((size, size), dtype=np.float)
    np.put_along_axis(sim, sim_indice, sim_matrix, 1)
    return sim, sim_matrix.shape


def load_sim_matrices(data_set, model_name_list):
    train_sims = []
    valid_sims = []
    test_sims = []
    data_set = data_set.split('/')[-1]

    for model_name in tqdm(model_name_list):
        train_sim_path = "./log/grid_search_%s_%s/train_sim.npy" % (model_name, data_set)
        train_sim = np.load(train_sim_path)
        train_sims.append(train_sim)
        valid_sim_path = "./log/grid_search_%s_%s/valid_sim.npy" % (model_name, data_set)
        valid_sim = np.load(valid_sim_path)
        valid_sims.append(valid_sim)
        test_sim_path = "./log/grid_search_%s_%s/test_sim.npy" % (model_name, data_set)
        test_sim = np.load(test_sim_path)
        test_sims.append(test_sim)
    return train_sims, valid_sims, test_sims


def generate_data(sims, ratio):
    assert sims[0].shape[0] == sims[0].shape[1]
    for i in range(1, len(sims)):
        assert sims[i].shape == sims[i - 1].shape
    sim_num = len(sims)
    size = sims[0].shape[0]
    sims = [np.reshape(sim, (size, size, 1)) for sim in sims]
    sims = np.concatenate(sims, axis=-1)  # shape = [size, size, sim_num]
    assert sims.shape == (size, size, sim_num)

    positive_data = [sims[i, i] for i in range(size)]
    negative_indice = np.random.randint(low=0, high=size, size=(ratio * size, 2))
    negative_indice = [(x, y) for x, y in negative_indice if x != y]

    negative_data = [sims[x, y] for x, y in negative_indice]
    data = positive_data + negative_data
    label = [1 for _ in range(len(positive_data))] + [0 for _ in range(len(negative_data))]
    data = [f.reshape(1, sim_num) for f in data]

    # shuffle
    tmp_box = list(zip(data, label))
    random.shuffle(tmp_box)
    data, label = zip(*tmp_box)

    data = np.concatenate(data, axis=0)
    label = np.asarray(label)
    return data, label


def ensemble_sims_with_svm(train_sims, valid_sims, test_sims, hits_1_weights, device, strategy='pre_weighted'):
    set_random_seed()

    def sim_standardization2(sim):
        mean = np.mean(sim)
        std = np.std(sim)
        sim = (sim - mean) / std
        return sim, mean, std

    def sim_standardization3(sim, mean, std):
        return (sim - mean) / std

    train_sims2 = []
    mean_list = []
    std_list = []
    for sim in train_sims:
        sim, mean, std = sim_standardization2(sim)
        train_sims2.append(sim)
        mean_list.append(mean)
        std_list.append(std)
    train_sims = train_sims2
    valid_sims = [sim_standardization3(sim, mean_list[i], std_list[i]) for i, sim in enumerate(valid_sims)]
    test_sims = [sim_standardization3(sim, mean_list[i], std_list[i]) for i, sim in enumerate(test_sims)]

    if strategy == 'avg':
        get_hits(sum(test_sims), device=device)
        return
    elif strategy == 'pre_weighted':
        channels_hits_1 = np.load(hits_1_weights)
        channels_weight = [i / channels_hits_1.sum() for i in channels_hits_1]
        sim_sum = channels_weight[0] * test_sims[0] + \
                  channels_weight[1] * test_sims[1] + \
                  channels_weight[2] * test_sims[2] + \
                  channels_weight[3] * test_sims[3]
        get_hits(sim_sum, device=device)
        return

    train_data, train_label = generate_data(train_sims, ratio=len(test_sims) * 4)
    test_data, test_label = generate_data(test_sims, ratio=1)

    def ensemble_sims_with_weight(test_sims, weight):
        # test performance
        test_size = test_sims[0].shape[0]
        test_sims = [sim.reshape(test_size, test_size, 1) for sim in test_sims]
        test_sims = np.concatenate(test_sims, axis=-1)
        test_sims = np.dot(test_sims, weight)
        test_sims = np.squeeze(test_sims, axis=-1)
        return - test_sims

    def performance_svc(train_data, train_label, test_sims, C):
        clf = SVC(kernel='linear', C=C, gamma='auto')
        clf.fit(train_data, train_label)
        prediction = clf.predict(test_data)
        print_time_info('Classification accuracy: %f.' % (np.sum(prediction == test_label) / len(test_label)))
        weight = clf.coef_.reshape(-1, 1)  # shape = [sim_num, 1]
        test_sims = ensemble_sims_with_weight(test_sims, weight)
        top_lr, top_rl, mr_lr, mr_rl, mrr_lr, mrr_rl = get_hits(test_sims, print_info=False, device=device)
        top1 = (top_lr[0] + top_rl[0]) / 2
        return top1, weight

    C_range = [1e-6, 1e-5]
    best_top1 = 0
    best_C = 0
    best_weight = None
    for C in C_range:
        top1, weight = performance_svc(train_data, train_label, valid_sims, C)
        if top1 > best_top1:
            best_top1 = top1
            best_C = C
            best_weight = weight
    test_sims = ensemble_sims_with_weight(test_sims, best_weight)
    print('Best C=%f.' % best_C)
    print('Weight', best_weight.reshape(-1))
    get_hits(test_sims, device=device)


def ensemble_partial_sim_matrix(data_set, device='cpu', strategy='pre_weighted'):
    def partial_get_hits(sim, top_k=(1, 10), print_info=True):
        if isinstance(sim, np.ndarray):
            sim = torch.from_numpy(sim)
        top_lr, mr_lr, mrr_lr = topk(sim, top_k, device=device)
        top_rl, mr_rl, mrr_rl = topk(sim.T, top_k, device=device)
        if print_info:
            print_time_info('For each source:', dash_top=True)
            print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_lr, mrr_lr))
            for i in range(len(top_lr)):
                print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_lr[i]))
            print_time_info('For each target:', dash_top=True)
            print_time_info('MR: %.2f; MRR: %.2f%%.' % (mr_rl, mrr_rl))
            for i in range(len(top_rl)):
                print_time_info('Hits@%d: %.2f%%' % (top_k[i], top_rl[i]))
        return top_lr, mr_lr, mrr_lr, top_rl, mr_rl, mrr_rl


    data_set = data_set.split('DWY100k/')[1]

    model_name_list = ['Literal', 'Digital', 'Structure', 'Name']
    if strategy == 'avg':
        sim_sum = []
        for model_name in tqdm(model_name_list):
            test_sim = np.load("./log/grid_search_%s_%s/test_sim.npy" % (model_name, data_set))
            if sim_sum == []:
                sim_sum = test_sim
            else:
                sim_sum += test_sim
        get_hits(sim_sum, device=device)
        return
    elif strategy == 'pre_weighted':
        sim_sum = []
        channels_hits_1 = np.load(hits_1_weights)
        channels_weight = [i / channels_hits_1.sum() for i in channels_hits_1]
        for i, model_name in tqdm(enumerate(model_name_list)):
            test_sim = np.load("./log/grid_search_%s_%s/test_sim.npy" % (model_name, data_set))
            if sim_sum == []:
                sim_sum = test_sim * channels_weight[i]
            else:
                sim_sum += test_sim * channels_weight[i]
        get_hits(sim_sum, device=device)
        return

    def svm_ensemble(train_sim_path_list, valid_sim_path_list, test_sim_path_list, T=True):
        positive_data = []  # shape = [sim_num, size]
        negative_data = []  # shape = [sim_num, size * ratio]
        sim_num = len(train_sim_path_list)

        size = 30000
        negative_indice = np.random.randint(low=0, high=size, size=(4 * sim_num * size, 2))
        negative_indice = [(x, y) for x, y in negative_indice if x != y]
        for sim_path in tqdm(train_sim_path_list, desc='Load train sims'):
            sim = np.load(sim_path)
            assert size == sim.shape[0]
            positive_data.append([sim[i, i] for i in range(size)])
            negative_data.append([sim[x, y] for x, y in negative_indice])

        positive_data = np.asarray(positive_data).T  # shape = [size, sim_num]
        negative_data = np.asarray(negative_data).T  # shape = [size * ratio, sim_num]
        print(positive_data.shape)
        print(negative_data.shape)

        valid_sims = []
        for sim_path in tqdm(valid_sim_path_list, desc='Load valid sims'):
            sim = np.load(sim_path)
            valid_sims.append(np.expand_dims(sim, -1))
        valid_sims = np.concatenate(valid_sims, axis=-1)  # shape = [size, size, sim_num]

        data = np.concatenate([positive_data, negative_data], axis=0)
        label = [1 for _ in range(len(positive_data))] + [0 for _ in range(len(negative_data))]
        label = np.asarray(label)

        C_range = [1e-6, 1e-5]
        best_C = 0
        best_top1 = 0
        best_weight = None
        for C in tqdm(C_range, desc='Fitting SVM'):
            clf = LinearSVC(random_state=0, C=C)
            clf.fit(data, label)
            weight = clf.coef_.reshape(-1, 1)
            tmp_valid_sims = np.dot(valid_sims, weight)
            tmp_valid_sims = np.squeeze(tmp_valid_sims, axis=-1)
            top_lr, mr_lr, mrr_lr, _, _, _ = partial_get_hits(-tmp_valid_sims, print_info=False)
            top1 = top_lr[0]
            if top1 > best_top1:
                best_top1 = top1
                best_weight = weight
                best_C = C
        print('Best C=%f' % best_C)
        print('Best weight', best_weight.reshape(-1))
        target_sim = None
        for idx, sim_path in tqdm(enumerate(test_sim_path_list), desc='Testing'):
            if target_sim is None:
                target_sim = best_weight[idx][0] * np.load(sim_path)
            else:
                target_sim += best_weight[idx][0] * np.load(sim_path)
        if T:
            target_sim = target_sim.T
        partial_get_hits(-target_sim)

    train_sim_path_list = ["./log/grid_search_%s_%s/train_sim.npy" % (model, data_set) for model in model_name_list]
    valid_sim_path_list = ["./log/grid_search_%s_%s/valid_sim.npy" % (model, data_set) for model in model_name_list]
    test_sim_path_list = ["./log/grid_search_%s_%s/test_sim.npy" % (model, data_set) for model in model_name_list]
    svm_ensemble(train_sim_path_list, valid_sim_path_list, test_sim_path_list)



if __name__ == '__main__':
    '''
    python ensemble_subgraphs.py --dataset DBP15k/zh_en
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default='0')
    parser.add_argument('--dataset', type=str, default='DBP15k/zh_en')  # DBP15k/zh_en, DBP15k/ja_en, DBP15k/fr_en, DWY100k/wd_dbp, DWY100k/yg_dbp
    parser.add_argument('--strategy', type=str, default='pre_weighted')  # avg, svm, pre_weighted
    parser.add_argument('--svm', action='store_true')
    args = parser.parse_args()
    hits_1_weights = str('log/channels_hits_1_' + args.dataset.split("/")[1] + '.npy')
    device = 'cuda:' + str(args.gpu_id)

    if args.dataset.find('DBP15k') >= 0:
        train_sims, valid_sims, test_sims = load_sim_matrices(args.dataset, ['Literal', 'Digital', 'Structure', 'Name'])
        ensemble_sims_with_svm(train_sims, valid_sims, test_sims, hits_1_weights, device=device, strategy=args.strategy)
    elif args.dataset.find('DWY100k') >= 0:
        ensemble_partial_sim_matrix(args.dataset, device=device, strategy=args.strategy)