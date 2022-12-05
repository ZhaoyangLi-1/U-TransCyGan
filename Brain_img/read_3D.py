import os
import torch
from tfrecord.torch.dataset import MultiTFRecordDataset
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import IterableDataset, ChainDataset, DataLoader, Dataset
import itertools 

class MyDataSet(Dataset):
    def __init__(self, data_path, index_path, length, len_tf):
        self.length = length
        self.data = []

        self.tfRecordDatasetList = []
        self.dataloaderList = []

        for i in range(len_tf):
            self.tfRecordDatasetList.append(TFRecordDataset(data_path[i], index_path[i]))
            
        for i in range(len_tf):
            self.dataloaderList.append(torch.utils.data.DataLoader(self.tfRecordDatasetList[i], batch_size=1))
            
        for i, dataloader in enumerate(self.dataloaderList):
            each_list = list(dataloader)
            self.data =  list(itertools.chain(self.data, each_list))
            
        del self.tfRecordDatasetList, self.dataloaderList
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return  self.data[index]


def data_loader(train_tfrecord_pattern, train_index_pattern,
                val_tfrecord_pattern, val_index_pattern, n_train,
                n_val, batch_size_train, batch_size_val):
    # train_tfrecord_pattern = "./Brain_img/3D/train/{}.tfrecords"
    # train_index_pattern = "./Brain_img/index/train/{}.index"
    # val_tfrecord_pattern = "./Brain_img/3D/validation/{}.tfrecords"
    # val_index_pattern = "./Brain_img/index/validation/{}.index"

    train_file_list = ["train{}".format(i) for i in [i for i in range(n_train)]]
    per_train = [float(1 / n_train)] * n_train
    splits_train = dict(zip(train_file_list, per_train))

    val_file_list = ["validation{}".format(i) for i in [i for i in range(n_val)]]
    per_val = [float(1 / n_val)] * n_val
    splits_val = dict(zip(val_file_list, per_val))

    train_dataset = MultiTFRecordDataset(train_tfrecord_pattern, train_index_pattern, splits_train, infinite=False)
    val_dataset = MultiTFRecordDataset(val_tfrecord_pattern, val_index_pattern, splits_val, infinite=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_val)

    return train_loader, val_loader


def tfrecords2index():
    train_file_list = ["~/CVLab/formal_project/Brain_img/3D/train/train{}.tfrecords".format(i) for i in [i for i in range(73)]]
    train_file_indexs = ["~/CVLab/formal_project/Brain_img/index/train/train{}.index".format(i) for i in [i for i in range(73)]]
    train_cmds = ["python -m tfrecord.tools.tfrecord2idx {} {}\n".format(train_file_list[i],
                                                                         train_file_indexs[i]) for i in range(73)]
    for cmd in train_cmds:
        os.system(cmd)

    val_file_list = ["~/CVLab/formal_project/Brain_img/3D/validation/validation{}.tfrecords".format(i) for i in [i for i in range(8)]]
    val_file_indexs = ["~/CVLab/formal_project/Brain_img/index/validation/validation{}.index".format(i) for i in [i for i in range(8)]]
    val_cmds = ["python -m tfrecord.tools.tfrecord2idx {} {}\n".format(val_file_list[i],
                                                                       val_file_indexs[i]) for i in range(8)]
    for cmd in val_cmds:
        os.system(cmd)


def get_data_loader_list(train_file_list, train_file_indices, val_file_list, val_file_indices, n_train, n_val):
    # train_file_list = ["./3D/train/train{}.tfrecords".format(i) for i in [i for i in range(73)]]
    # train_file_indices = ["./index/train/train{}.index".format(i) for i in [i for i in range(73)]]
    # val_file_list = ["./3D/validation/validation{}.tfrecords".format(i) for i in  [i for i in range(8)]]
    # val_file_indices = ["./index/validation/validation{}.index".format(i) for i in  [i for i in range(8)]]
    train_dataset_list = []
    val_dataset_list = []
    for i in range(n_train):
        dataset = MyDataSet(train_file_list[i], train_file_indices[i], 1)
        train_dataset_list.append(dataset)

    for i in range(n_val):
        dataset = MyDataSet(val_file_list[i], val_file_indices[i], 1)
        val_dataset_list.append(dataset)

    return train_dataset_list, val_dataset_list


def concat_dataset_loader(train_dataset_list, val_dataset_list, batch_train, batch_val):
    train_dataset = ChainDataset(train_dataset_list)
    val_dataset = ChainDataset(val_dataset_list)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_train, drop_last=True,
                              sampler=InfiniteSamplerWrapper(train_dataset),
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_val, drop_last=True,
                             sampler=InfiniteSamplerWrapper(val_dataset),
                             num_workers=0)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_train, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_val, drop_last=True)
    return train_loader, val_loader


def get_data_loader(args):
    train_file_list = [args.train_tfrecord_pattern.format(i) for i in [i for i in range(73)]]
    train_file_indices = [args.train_index_pattern.format(i) for i in [i for i in range(73)]]
    val_file_list = [args.val_tfrecord_pattern.format(i) for i in [i for i in range(8)]]
    val_file_indices = [args.val_index_pattern.format(i) for i in [i for i in range(8)]]
    # train_dataset_list, val_dataset_list = \
    #     get_data_loader_list(train_file_list, train_file_indices,
    #                          val_file_list, val_file_indices, 73, 8)
    # train_loader, val_loader = concat_dataset_loader(train_dataset_list, val_dataset_list, args.batch_size, args.batch_size)
    train_dataset = MyDataSet(train_file_list, train_file_indices, args.n_train, 73)
    val_dataset = MyDataSet(val_file_list, val_file_indices, args.n_val, 8)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    # list_train = list(train_loader)

    return train_loader, val_loader


def get_val_loader(args):
    _, val_loader = get_data_loader(args)
    return val_loader


def get_train_loader(args):
    train_loader, _ = get_data_loader(args)
    return train_loader


# def main():
#     device = torch.device("cuda:0")
#     train_tfrecord_pattern = "./3D/train/{}.tfrecords"
#     train_index_pattern = "./index/train/{}.index"
#     val_tfrecord_pattern = "./3D/validation/{}.tfrecords"
#     val_index_pattern = "./index/validation/{}.index"
#     train_file_list = ["./3D/train/train{}.tfrecords".format(i) for i in [i for i in range(73)]]
#     train_file_indices = ["./index/train/train{}.index".format(i) for i in [i for i in range(73)]]
#     val_file_list = ["./3D/validation/validation{}.tfrecords".format(i) for i in [i for i in range(8)]]
#     val_file_indices = ["./index/validation/validation{}.index".format(i) for i in [i for i in range(8)]]
#     train_dataset_list, val_dataset_list = \
#         get_data_loader_list(train_file_list, train_file_indices, val_file_list, val_file_indices, 73, 8)
#     train_loader, val_loader = concat_dataset_loader(train_dataset_list, val_dataset_list, 4, 4)
#     a=1
#   tfrecords2index()

# if __name__ == "__main__":
#     main()
