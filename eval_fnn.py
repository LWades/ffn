
from utils import log

import torch

from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import h5py
from args import args
from train import valid_eval
from network import get_model
import numpy as np
from data_process import ToricDataset


key_syndrome = 'syndromes'
key_logical_error = 'logical_errors'
pwd_trndt = '/root/Surface_code_and_Toric_code/{}_pe/'.format(args.c_type)
pwd_model = '/root/ffn/output/'
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

ps = torch.linspace(0.01, 0.20, 20)

model = get_model()

accs = np.zeros(20)
i = 0

log("Eval...")

for p in ps:
    filename_test_data = pwd_trndt + '{}_d{}_p{}_trnsz{}_eval_seed{}.hdf5'.format(args.c_type, args.d, format(p, '.3f'), 10000, args.eval_seed)
    log("test_data: {}".format(filename_test_data))
    with h5py.File(filename_test_data, 'r') as f:
        test_syndrome = f[key_syndrome][()]
        test_logical_error = f[key_logical_error][()]
        testset = ToricDataset({key_syndrome: test_syndrome, key_logical_error: test_logical_error})

    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=2,
                             pin_memory=True) if testset is not None else None

    log("test_loader.dataset type: {}".format(type(test_loader.dataset)))
    log("test_loader.dataset: {}".format(test_loader.dataset))

    model_name = '{}_checkpoint.bin'.format(args.name)
    # model_name = 'fnn_{}_{}-5e6_checkpoint.bin'.format(args.d, format(args.p, '.2f'))
    log("model: {}".format(model_name))
    model.load_state_dict(torch.load(pwd_model + model_name))
    model.to(device)
    model.eval()

    acc = valid_eval(model, test_loader)
    accs[i] = acc
    i += 1
    log("p {} acc: {}".format(format(p, '.3f'), acc))
log("accs: \n{}".format(accs))
log("Eval... Done.")
# python3 eval_plot.py --c_type sur --d 11 --name sur-11-0.10-1e7
# python3 eval_plot.py --name sur-11-0.10-1e7 --dataset sur --model_type Sur_11 --d 11 --p 0.10 --img_size 21 --eval_seed 1 --fp16 --fp16_opt_level O2
# python3 eval_plot.py --name sur-11-0.10-1e7 --dataset sur --model_type Sur_11 --d 11 --p 0.10 --img_size 21 --eval_seed 1 --fp16 --fp16_opt_level O2
# python3 eval_plot.py --dataset torc --model_type Sur_11 --d 11 --p 0.10 --img_size 21 --eval_seed 1 --fp16 --fp16_opt_level O2
# python3 eval_fnn.py --nn fnn --c_type torc --d 5 --k 2 --p 0.10 --eval_seed 1
# nohup python3 eval_fnn.py --nn fnn --c_type torc --d 5 --k 2 --p 0.10 --eval_seed 1 > logs/ef.log &
# nohup python3 eval_fnn.py --nn fnn --c_type torc --d 5 --k 2 --p 0.07 --eval_seed 1 > logs/ef.log &
