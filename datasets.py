import copy

import math
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import mindspore.dataset as ds


class IMVDataset:
    def __init__(self, imv_data, mask_matrix, labels, num_views):
        self.num_views = num_views
        self.imv_data = imv_data
        self.mask = mask_matrix
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        items = [dv[index] for dv in self.imv_data]
        items.append(self.mask[index])
        items.append(self.labels[index])
        return items


class SVDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


def get_mask(num_views, data_size, missing_rate):
    assert num_views >= 2
    miss_sample_num = math.floor(data_size * missing_rate)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, num_views])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(num_views)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[miss_ind[j]] = rand_v
    return mask


def load_data(args):
    data_path = args.dataset_dir_base + args.dataset_name + '.npz'
    data = np.load(data_path)
    num_views = int(data['n_views'])
    origin_mv_data = [data[f'view_{v}'].astype(np.float32) for v in range(num_views)]
    labels = data['labels'].astype(np.float32)
    data_size = labels.shape[0]
    dims = [sv_data.shape[1] for sv_data in origin_mv_data]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = num_views
    args.class_num = class_num
    args.data_size = data_size
    return origin_mv_data, labels


def pixel_normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def build_dataset(args):
    origin_mv_data, labels = load_data(args)

    if args.dataset_name == 'Caltech7-5V':
        origin_mv_data = [pixel_normalize(sv_data) for sv_data in origin_mv_data]
    elif args.dataset_name == 'Scene-15':
        origin_mv_data = [StandardScaler().fit_transform(sv_data) for sv_data in origin_mv_data]
    else:
        pass

    mask = get_mask(args.num_views, args.data_size, args.missing_rate).astype(np.float32)
    imv_data = [origin_mv_data[v] * mask[:, v:v + 1] for v in range(args.num_views)]
    items_name = [f'view_{v}' for v in range(args.num_views)]
    items_name.extend(['mask', 'labels'])

    imv_dataset = ds.GeneratorDataset(IMVDataset(imv_data, mask, labels, args.num_views), column_names=items_name, shuffle=True)
    imv_dataset = imv_dataset.batch(args.batch_size, drop_remainder=False)
    imv_loader = imv_dataset.create_tuple_iterator()

    com_idx = np.sum(mask, axis=1) == args.num_views
    cmv_data = [sv_data[com_idx] for sv_data in imv_data]

    sv_datasets = [ds.GeneratorDataset(SVDataset(copy.deepcopy(imv_data[v][mask[:, v] == 1])), column_names=['data'], shuffle=True) for v in range(args.num_views)]
    sv_datasets = [sv_dataset.batch(args.batch_size, drop_remainder=False) for sv_dataset in sv_datasets]
    sv_loaders = [sv_dataset.create_tuple_iterator() for sv_dataset in sv_datasets]

    return cmv_data, imv_loader, sv_loaders
