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
                if len(self.s_types) > 0:
                    for s_type_other in self.s_types:
                        if s_type_other == 'tl':
                            continue
                        bs = copy.deepcopy(symmetric_syndrome)
                        bs.syndrome, bs_xs, bs_ys = bs.base_syndrome_xs_ys()
                        bs.xs, bs.ys, bs.center = bs_xs, bs_ys, bs.center_img
                        if s_type_other == 'rf:0':
                            bs.syndrome = bs.reflection_syndrome(0)
                        if s_type_other == 'rf:1':
                            bs.syndrome = bs.reflection_syndrome(1)
                        if s_type_other == 'rf:2':
                            bs.syndrome = bs.reflection_syndrome(2)
                        if s_type_other == 'rf:3':
                            bs.syndrome = bs.reflection_syndrome(3)
                        if s_type_other == 'rt:0':
                            bs.syndrome = bs.rotation_syndrome(0)
                        if s_type_other == 'rt:1':
                            bs.syndrome = bs.rotation_syndrome(1)
                        if s_type_other == 'rt:2':
                            bs.syndrome = bs.rotation_syndrome(2)
                        bs.xs = np.where(bs.syndrome == 1)[0]
                        bs.ys = np.where(bs.syndrome == 1)[1]
                        bs.center = bs.get_center()
                        if bs not in self.symmetric_dicts:
                            self.symmetric_dicts[bs] = 1
                        elif self.symmetric_dicts[bs] < self.rep_limit:
                            self.symmetric_dicts[bs] += 1
                    break
                # for s_type02 in self.s_types:
                #     if s_type02 == 'tl':
                #         continue
            else:
                bs = copy.deepcopy(symmetric_syndrome)
                if s_type == 'rf:0':
                    bs.syndrome = symmetric_syndrome.reflection_syndrome(0)
                elif s_type == 'rf:1':
                    bs.syndrome = symmetric_syndrome.reflection_syndrome(1)
                elif s_type == 'rf:2':
                    bs.syndrome = symmetric_syndrome.reflection_syndrome(2)
                elif s_type == 'rf:3':
                    bs.syndrome = symmetric_syndrome.reflection_syndrome(3)
                elif s_type == 'rt:0':
                    bs.syndrome = symmetric_syndrome.rotation_syndrome(0)
                elif s_type == 'rt:1':
                    bs.syndrome = symmetric_syndrome.rotation_syndrome(1)
                elif s_type == 'rt:2':
                    bs.syndrome = symmetric_syndrome.rotation_syndrome(2)
                if bs not in self.symmetric_dicts:
                    self.symmetric_dicts[bs] = 1
                elif self.symmetric_dicts[bs] < self.rep_limit:
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
    file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_seed{}.hdf5".format(args.c_type,
                                                                                                args.c_type,
                                                                                                args.d,
                                                                                                format(args.p, '.3f'),
                                                                                                args.trnsz, args.seed)
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
    file_name = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_seed{}.hdf5".format(args.c_type,
                                                                                                args.c_type,
                                                                                                args.d,
                                                                                                format(args.p, '.3f'),
                                                                                                args.poolsz, args.seed)
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
    file_name_zip = "/root/Surface_code_and_Toric_code/{}_pe_zip/{}_d{}_p{}_trnsz{}_limit{}_seed{}".format(args.c_type,
                                                                                                           args.c_type,
                                                                                                           args.d,
                                                                                                           format(
                                                                                                               args.p,
                                                                                                               '.3f'),
                                                                                                           args.trnsz,
                                                                                                           args.limit,
                                                                                                           args.seed)
    for s_type in s_types:
        file_name_zip = file_name_zip + "_" + s_type
    file_name_zip = file_name_zip + ".hdf5"
    with h5py.File(file_name_zip, 'w') as f:
        f.create_dataset('syndromes', data=syndromes_zip)
        f.create_dataset('logical_errors', data=logical_errors_zip)
    log("zip finish. File name zip: {}".format(file_name_zip))
    return num_sum_repeat, num_max_repeat


if __name__ == '__main__':
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

# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.010 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_17.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.050 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_18.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.150 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_25.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.050 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_19.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.010 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_24.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.150 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_26.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.100 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_22.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.100 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_23.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.010 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_27.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.010 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_28.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.050 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_29.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.050 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_44.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.100 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_31.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.100 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_32.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.150 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_33.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.150 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_34.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.150 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_35.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.010 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_36.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.010 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_37.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.050 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_38.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.050 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_39.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.100 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_40.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.100 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_41.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.150 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_42.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.150 --trnsz 10000000 --sym 'rf:0' > logs/symmetric_filter_43.log &
# 0.07
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.070 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_50.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.070 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_51.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.070 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_52.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.070 --trnsz 10000000 --sym 'tl' > logs/symmetric_filter_53.log &
# rf
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.050 --trnsz 10000000 --sym 'rf:0' 'rf:1' 'rf:2' 'rf:3' > logs/symmetric_filter_54.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.050 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_55.log &

# 3 0.15
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.150 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_56.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.150 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_57.log &

# 7 0.01-0.15
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.010 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_58.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.010 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_59.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.050 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_60.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.050 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_61.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.100 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_62.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.100 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_63.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.150 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_64.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.150 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_65.log &

# 9 0.01-0.15
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.010 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_66.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.010 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_67.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.050 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_68.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.050 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_69.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.100 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_70.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.070 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_74.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.100 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_71.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.150 --trnsz 10000000 --sym 'rf' > logs/symmetric_filter_72.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.150 --trnsz 10000000 --sym 'rt' > logs/symmetric_filter_73.log &

# all
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 3 --p 0.010 --trnsz 10000000 --sym 'all' > logs/symmetric_filter_80.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.010 --trnsz 10000000 --sym 'all' > logs/symmetric_filter_81.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 5 --p 0.050 --trnsz 10000000 --sym 'all' > logs/symmetric_filter_84.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.010 --trnsz 10000000 --sym 'all' > logs/symmetric_filter_82.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 7 --p 0.070 --trnsz 10000000 --sym 'all' > logs/symmetric_filter_85.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.010 --trnsz 10000000 --sym 'all' > logs/symmetric_filter_83.log &
# nohup python3 symmetric_filter.py --zip 0 --c_type torc --d 9 --p 0.070 --trnsz 10000000 --sym 'all' > logs/symmetric_filter_86.log &
# nohup python3 symmetric_filter.py --zip 1 --c_type torc --d 7 --p 0.070 --limit 100000 --trnsz 5000000 --poolsz 10000000 --sym 'all' > logs/symmetric_filter_87.log &

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
# syndrome02 = np.array([0, 1, 1,
#                        0, 0, 0,
#                        0, 0, 0,
#                        1, 0, 0,
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
