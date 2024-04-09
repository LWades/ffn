from args import args
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils import log
import numpy as np
import h5py

path_data = '/root/Surface_code_and_Toric_code/'


filename_test_data = '{}_pe/{}_d{}_p{}_trnsz{}_eval_seed1.hdf5'.format(args.c_type, args.c_type, args.d, format(args.p, '.3f'), args.testsz)


if args.zip == 1:
    filename_train_data = "{}_pe_zip/{}_d{}_p{}_trnsz{}_limit{}_seed{}".format(args.c_type, args.c_type,
                                                                                         args.d, format(args.p, '.3f'), args.trnsz, args.limit, args.seed)
    for s_type in args.sym:
        filename_train_data = filename_train_data + "_" + s_type
else:
    filename_train_data = '{}_pe/{}_d{}_p{}_trnsz{}'.format(args.c_type, args.c_type, args.d,
                                                                       format(args.p, '.3f'), args.trnsz)

if args.nn == 'cnn':
    filename_train_data = filename_train_data + "_imgsdr"
    filename_test_data = '{}_pe/{}_d{}_p{}_trnsz{}_imgsdr_eval_seed1.hdf5'.format(args.c_type, args.c_type, args.d,
                                                                           format(args.p, '.3f'), args.testsz)

if args.zip != 1:
    filename_train_data = filename_train_data + "_seed{}.hdf5".format(args.seed)
else:
    filename_train_data = filename_train_data + ".hdf5".format(args.seed)
# /root/qecGPT/qec/trndt/toricode/torc_d3_0.010_trnsz10000000.npz


class ToricDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['syndromes'])

    def __getitem__(self, idx):
        syndrome_data = self.data['syndromes'][idx]
        logical_error_data = self.data['logical_errors'][idx]
        # log("1 {}".format(syndrome_data))
        # log("2 {}".format(logical_error_data))
        return syndrome_data, logical_error_data


class SurImgDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['image_syndromes'])

    def __getitem__(self, idx):
        syndrome_data = self.data['image_syndromes'][idx]
        logical_error_data = self.data['logical_errors'][idx]
        # log("1 {}".format(syndrome_data))
        # log("2 {}".format(logical_error_data))
        return syndrome_data, logical_error_data


class SurDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['image_syndromes'])

    def __getitem__(self, idx):
        return self.data['image_syndromes'][idx], self.data['logical_errors'][idx]


def get_loader():
    log("Data loading...")
    # traindata = np.load(path_data + filename_train_data)
    # traindata = np.load('/root/qecGPT/qec/trndt/sur_d3_p0.010_trnsz10000000_imgsdr.npz')
    # testdata = np.load('/root/qecGPT/qec/trndt/sur_d3_p0.010_trnsz10000_imgsdr_eval.npz')
    # testdata = np.load(path_data + filename_test_data)
    log("traindata: {}".format(path_data + filename_train_data))
    with h5py.File(path_data + filename_train_data, 'r') as f:
        train_syndrome = f['syndromes'][()]
        log("traindata size: {}".format(train_syndrome.shape[0]))
        train_logical_error = f['logical_errors'][()]
        trainset = ToricDataset({'syndromes': train_syndrome, 'logical_errors': train_logical_error})
    log("testdata: {}".format(path_data + filename_test_data))
    with h5py.File(path_data + filename_test_data, 'r') as f:
        test_syndrome = f['syndromes'][()]
        log("testdata size: {}".format(test_syndrome.shape[0]))
        test_logical_error = f['logical_errors'][()]
        testset = ToricDataset({'syndromes': test_syndrome, 'logical_errors': test_logical_error})
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             pin_memory=True) if testset is not None else None
    log("Data loading... Done.")
    return train_loader, test_loader


def get_loader_sur_img():
    log("Data loading...")
    # traindata = np.load(path_data + filename_train_data)
    # traindata = np.load('/root/qecGPT/qec/trndt/sur_d3_p0.010_trnsz10000000_imgsdr.npz')
    # testdata = np.load('/root/qecGPT/qec/trndt/sur_d3_p0.010_trnsz10000_imgsdr_eval.npz')
    # testdata = np.load(path_data + filename_test_data)
    log("traindata: {}".format(path_data + filename_train_data))
    with h5py.File(path_data + filename_train_data, 'r') as f:
        train_syndrome = f['image_syndromes'][()]
        log("traindata size: {}".format(train_syndrome.shape[0]))
        train_syndrome_post = np.expand_dims(train_syndrome, axis=1)
        train_logical_error = f['logical_errors'][()]
        trainset = SurImgDataset({'image_syndromes': train_syndrome_post, 'logical_errors': train_logical_error})
    log("testdata: {}".format(path_data + filename_test_data))
    with h5py.File(path_data + filename_test_data, 'r') as f:
        test_syndrome = f['image_syndromes'][()]
        log("testdata size: {}".format(test_syndrome.shape[0]))
        test_syndrome_post = np.expand_dims(test_syndrome, axis=1)
        test_logical_error = f['logical_errors'][()]
        testset = SurImgDataset({'image_syndromes': test_syndrome_post, 'logical_errors': test_logical_error})
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             pin_memory=True) if testset is not None else None
    log("Data loading... Done.")
    return train_loader, test_loader