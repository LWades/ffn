from args import args
from symmetric_syndrome import ZX2image, SymmetricSyndrome
from utils import log
import h5py
import wandb
from tqdm import tqdm
import numpy as np
import copy

"""
s_types ->
1. tl
2. rf
"""


class SymmetricFilter:
    def __init__(self, d, s_types, rep_limit):
        self.d = d
        self.s_types = s_types  # 应该是一个集合，其中可以不仅包含一种对称性
        self.symmetric_dicts = {}  # 存症状的地方
        # self.symmetric_dicts = [{} for _ in range(7)]  # 存症状的地方
        self.rep_limit = rep_limit  # 允许重复的上限；压缩系数
        self.record = {}

    def add(self, symmetric_syndrome):
        for s_type in self.s_types:
            if s_type == 'tl':
                if symmetric_syndrome not in self.symmetric_dicts:
                    self.symmetric_dicts[symmetric_syndrome] = 1
                elif self.symmetric_dicts[symmetric_syndrome] < self.rep_limit:
                    self.symmetric_dicts[symmetric_syndrome] += 1
            elif s_type == 'rf:0':
                bs = copy.deepcopy(symmetric_syndrome)
                bs.syndrome = symmetric_syndrome.reflection_syndrome(0)
                if bs not in self.symmetric_dicts:
                    self.symmetric_dicts[bs] = 1
                elif self.symmetric_dicts[symmetric_syndrome] < self.rep_limit:
                    self.symmetric_dicts[bs] += 1
            elif s_type == 'rf:1':
                bs = copy.deepcopy(symmetric_syndrome)
                bs.syndrome = symmetric_syndrome.reflection_syndrome(1)
                if bs not in self.symmetric_dicts:
                    self.symmetric_dicts[bs] = 1
                elif self.symmetric_dicts[symmetric_syndrome] < self.rep_limit:
                    self.symmetric_dicts[bs] += 1
            elif s_type == 'rf:2':
                bs = copy.deepcopy(symmetric_syndrome)
                bs.syndrome = symmetric_syndrome.reflection_syndrome(2)
                if bs not in self.symmetric_dicts:
                    self.symmetric_dicts[bs] = 1
                elif self.symmetric_dicts[symmetric_syndrome] < self.rep_limit:
                    self.symmetric_dicts[bs] += 1
            elif s_type == 'rf:3':
                bs = copy.deepcopy(symmetric_syndrome)
                bs.syndrome = symmetric_syndrome.reflection_syndrome(3)
                if bs not in self.symmetric_dicts:
                    self.symmetric_dicts[bs] = 1
                elif self.symmetric_dicts[symmetric_syndrome] < self.rep_limit:
                    self.symmetric_dicts[bs] += 1

    """
    如果存在重复并次数超过了rep_limit，就返回 False，表示抛弃这个错误症状，否则为 True，加入训练集
    """

    def filter(self, syndrome):
        symmetric_syndrome = SymmetricSyndrome(self.d, syndrome, self.s_types)
        num = 1
        if symmetric_syndrome in self.symmetric_dicts:
            num = self.symmetric_dicts[symmetric_syndrome]
            if num >= rep_limit:
                # log("translation: exist in dict and exceed the limit")
                return False, num
        # 过了这么多关，可以加入到训练集了，把它所有对称的都加进去
        self.add(symmetric_syndrome)
        return True, num


    # return False 有
    # return True 没有
    def filter_eval(self, syndrome):
        num = 1
        symmetric_syndrome = SymmetricSyndrome(self.d, syndrome, self.s_types)
        if symmetric_syndrome in self.symmetric_dicts:
            num = self.symmetric_dicts[symmetric_syndrome]
            if num >= rep_limit:
                return False, num
            else:
                self.add(symmetric_syndrome)
                return False, num
        else:
            self.add(symmetric_syndrome)
            return True, num

    def get_size(self):
        return self.symmetric_dicts.__len__()



def get_data(file_name):
    with h5py.File(file_name, 'r') as f:
        syndromes = f['syndromes'][()]
        logical_errors = f['logical_errors'][()]
        return syndromes, logical_errors


s_types, d, rep_limit, trnsz = args.sym, args.d, args.limit, args.trnsz


def sym_eval():
    num_max_repeat = 0
    num_sum_repeat = 0
    file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_seed{}.hdf5".format(args.c_type, args.c_type,
                                                                                         args.d, format(args.p, '.3f'), args.trnsz, args.seed)
    symmetric_filter = SymmetricFilter(d, s_types, rep_limit)
    syndromes, logical_errors = get_data(file_name)
    for i in tqdm(range(len(syndromes))):
        result, num_repeat = symmetric_filter.filter_eval(syndromes[i])
        if not result:
            num_sum_repeat += 1
            if num_repeat > num_max_repeat:
                num_max_repeat = num_repeat
        print(num_sum_repeat, end='\r')
    return num_sum_repeat, num_max_repeat


def sym_zip():
    num_max_repeat = 0
    num_sum_repeat = 0
    symmetric_filter = SymmetricFilter(d, s_types, rep_limit)
    file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_seed{}.hdf5".format(args.c_type, args.c_type,
                                                                                         args.d, format(args.p, '.3f'), args.poolsz, args.seed)
    syndromes, logical_errors = get_data(file_name)
    syndromes_zip, logical_errors_zip = [], []
    count = 0
    for i in tqdm(range(len(syndromes))):
        if count >= trnsz:
            break
        result, num_repeat = symmetric_filter.filter(syndromes[i])
        if not result:
            num_sum_repeat += 1
            if num_repeat > num_max_repeat:
                num_max_repeat = num_repeat
        else:
            syndromes_zip.append(syndromes[i])
            logical_errors_zip.append(logical_errors[i])
            count += 1
    file_name_zip = "/root/Surface_code_and_Toric_code/{}_pe_zip/{}_d{}_p{}_trnsz{}_limit{}_seed{}".format(args.c_type, args.c_type,
                                                                                         args.d, format(args.p, '.3f'), args.trnsz, args.limit, args.seed)
    for s_type in s_types:
        file_name_zip = file_name_zip + "_" + s_type
    file_name_zip = file_name_zip + ".hdf5"
    with h5py.File(file_name_zip, 'w') as f:
        f.create_dataset('syndromes', data=syndromes_zip)
        f.create_dataset('logical_errors', data=logical_errors_zip)
    log("zip finish. File name zip: {}".format(file_name_zip))
    return num_sum_repeat, num_max_repeat



if __name__ == '__main__':
    # wandb.init(
    #     project="symmetry",
    #     name=f"d{args.d}_p{args.p}_trnsz{args.trnsz}_zip{args.zip}",
    #     config={
    #         'd': args.d,
    #         'p': args.p,
    #         'train size': args.trnsz,
    #     }
    # )
    # log_table = wandb.Table(columns=["num_sum_repeat", "num_max_repeat"])
    if args.zip == 1:
        log("symmetry zip...")
        num_sum_repeat, num_max_repeat = sym_zip()
        log("symmetry zip... Done.")
    else:
        log("symmetry eval...")
        num_sum_repeat, num_max_repeat = sym_eval()
        log("symmetry eval... Done.")
    log("num_sum_repeat: {}".format(num_sum_repeat))
    log("num_max_repeat: {}".format(num_max_repeat))
    # wandb.log({"sym_info": wandb.plot.bar(log_table, "num_sum_repeat", "num_max_repeat", title="sym_info")})
    # wandb.finish()

# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.040 --trnsz 5000000 --sym 'tl' > logs/symmetric_filter_1.log &
# nohup python3 symmetric_filter.py --zip 1 --limit 4000 --c_type torc --d 5 --p 0.040 --trnsz 5000000 --sym 'tl' > logs/symmetric_filter_3.log &
# nohup python3 symmetric_filter.py --zip 1 --limit 1000 --c_type torc --d 5 --p 0.040 --trnsz 5000000 --sym 'tl' > logs/symmetric_filter_10.log &
# nohup python3 symmetric_filter.py --zip 1 --limit 3000 --c_type torc --d 5 --p 0.010 --trnsz 5000000 --sym 'tl' > logs/symmetric_filter_11.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.010 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_5.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.010 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_7.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.050 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_6.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.010 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_8.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.050 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_9.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.100 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_11.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.150 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_10.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.010 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_12.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.050 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_13.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.100 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_14.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.150 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_16.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.010 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_15.log &

# symmetric_filter = SymmetricFilter(d, s_types, rep_limit)
#
# syndrome01 = np.array([0, 1, 0,
#                        0, 0, 1,
#                        0, 0, 0,
#                        0, 0, 1,
#                        0, 0, 1,
#                        0, 0, 0])
# syndrome01 = syndrome01[1:-1]
# log(syndrome01)
# # syndrome02 = np.array([0, 0, 1,
# #                        0, 1, 0,
# #                        0, 0, 0,
# #                        0, 0, 1,
# #                        0, 0, 1,
# #                        0, 0, 0])
# # syndrome02 = np.array([0, 1, 0,
# #                        0, 0, 0,
# #                        0, 0, 1,
# #                        0, 0, 1,
# #                        0, 0, 1,
# #                        0, 0, 0])
# # syndrome02 = np.array([0, 1, 1,
# #                        0, 0, 0,
# #                        0, 0, 0,
# #                        0, 0, 1,
# #                        1, 0, 0,
# #                        0, 0, 0])
# syndrome02 = np.array([0, 1, 1,
#                        0, 0, 0,
#                        0, 0, 0,
#                        0, 1, 0,
#                        0, 0, 1,
#                        0, 0, 0])
# syndrome02 = syndrome02[1:-1]
# log(syndrome02)
# f1 = symmetric_filter.filter(syndrome01)
#
# log(f1)
# log(symmetric_filter.get_size())
#
# f2 = symmetric_filter.filter(syndrome02)
#
# log(f2)
# log(symmetric_filter.get_size())

