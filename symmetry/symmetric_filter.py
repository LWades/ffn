from args import args
from symmetric_syndrome import ZX2image
from utils import log
import h5py
import wandb
import numpy as np

"""
s_types ->
1. tl
2. rf
"""


class SymmetricFilter:
    def __init__(self, d, s_types, rep_limit):
        self.d = d
        self.s_types = s_types  # 应该是一个集合，其中可以不仅包含一种对称性
        self.symmetric_dict = {}  # 存症状的地方，一个大字典
        self.rep_limit = rep_limit  # 允许重复的上限；压缩系数
        self.record = {}

    """
    如果存在重复并次数超过了rep_limit，就返回 False，表示抛弃这个错误症状，否则为 True，加入训练集
    """

    def filter(self, syndrome):
        symmetric_syndrome = ZX2image(self.d, syndrome)
        # for s_type in self.s_types:
        #     if s_type == 'rf':
        num = 1
        if symmetric_syndrome in self.symmetric_dict:
            self.symmetric_dict[symmetric_syndrome] += 1
            num = self.symmetric_dict[symmetric_syndrome]
            if self.symmetric_dict[symmetric_syndrome] >= self.rep_limit:
                return False, num
        else:
            self.symmetric_dict[symmetric_syndrome] = 1
        return True, num


def get_data(file_name):
    with h5py.File(file_name, 'r') as f:
        syndromes = f['syndromes'][()]
        logical_errors = f['logical_errors'][()]
        return syndromes, logical_errors


s_types, d, rep_limit, trnsz = args.sym, args.d, args.limit, args.trnsz


def sym_eval():
    num_max_repeat = 0
    num_sum_repeat = 0
    file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}.hdf5".format(args.c_type, args.c_type,
                                                                                         args.d, format(args.p, '.3f'), args.trnsz)
    symmetric_filter = SymmetricFilter(d, s_types, rep_limit)
    syndromes, logical_errors = get_data(file_name)
    for i in range(len(syndromes)):
        result, num_repeat = symmetric_filter.filter(syndromes[i])
        if not result:
            num_sum_repeat += 1
            if num_repeat > num_max_repeat:
                num_max_repeat = num_repeat
    return num_sum_repeat, num_max_repeat


def sym_zip():
    num_max_repeat = 0
    num_sum_repeat = 0
    symmetric_filter = SymmetricFilter(d, s_types, rep_limit)
    file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}.hdf5".format(args.c_type, args.c_type,
                                                                                         args.d, format(args.p, '.3f'), args.poolsz)
    syndromes, logical_errors = get_data(file_name)
    syndromes_zip, logical_errors = [], []
    for i in range(len(syndromes)):
        if i >= trnsz:
            break
        result, num_repeat = symmetric_filter.filter(syndromes[i])
        if not result:
            num_sum_repeat += 1
            if num_repeat > num_max_repeat:
                num_max_repeat = num_repeat
        else:
            syndromes_zip.append(syndromes[i])
            logical_errors.append(logical_errors[i])
    file_name_zip = "/root/Surface_code_and_Toric_code/{}_pe_zip/{}_d{}_p{}_trnsz{}_limit{}.hdf5".format(args.c_type, args.c_type,
                                                                                         args.d, format(args.p, '.3f'), args.trnsz, args.limit)
    with h5py.File(file_name_zip, 'w') as f:
        f.create_dataset('syndromes', data=syndromes_zip)
        f.create_dataset('logical_errors', data=logical_errors)
    log("zip finish. File name zip: {}".format(file_name_zip))
    return num_sum_repeat, num_max_repeat


if __name__ == '__main__':
    wandb.init(
        project="symmetry",
        name=f"d{args.d}_p{args.p}_trnsz{args.trnsz}_zip{args.zip}",
        config={
            'd': args.d,
            'p': args.p,
            'train size': args.trnsz,
        }
    )
    log_table = wandb.Table(columns=["num_sum_repeat", "num_max_repeat"])
    if args.zip == 1:
        num_sum_repeat, num_max_repeat = sym_zip()
    else:
        num_sum_repeat, num_max_repeat = sym_eval()
    wandb.log({"sym_info": wandb.plot.bar(log_table, "num_sum_repeat", "num_max_repeat", title="sym_info")})
    wandb.finish()

# python3 symmetry/symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.010 --trnsz 10000000 --sym ['tl']
