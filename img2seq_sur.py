from args import args
from utils import log
import h5py
import numpy as np
from tqdm import tqdm

l = 2 * args.d - 1

def imgsdr2ZX(image_syndrome):
    syndrome = np.zeros(2 * args.d ** 2 - 2 * args.d)
    k = 0
    for i in range(l):
        for j in range(l):
            if i % 2 == 0 and j % 2 == 1:
                syndrome[k] = 1 if image_syndrome[i, j] == -1 else 0
                k += 1
            if i % 2 == 1 and j % 2 == 0:
                syndrome[k] = 1 if image_syndrome[i, j] == -1 else 0
                k += 1
    return syndrome


path_data = '/root/Surface_code_and_Toric_code/{}_pe/'.format(args.c_type)
path_data_base = '/root/Surface_code_and_Toric_code/{}_base_pe/'.format(args.c_type)
filename_read_data = '{}_d{}_p{}_trnsz{}_imgsdr_seed0.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), args.trnsz)
filename_write_data = '{}_d{}_p{}_trnsz{}_seed0.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), args.trnsz)

log("img2seq...")
with h5py.File(path_data + filename_read_data, 'r') as f:
    image_syndromes = f['image_syndromes'][()]
    logical_errors = f['logical_errors'][()]
    num = image_syndromes.shape[0]
with h5py.File(path_data_base + filename_write_data, 'w') as f:
    syndromes = np.zeros((args.trnsz, 2 * args.d ** 2 - 2 * args.d))
    for i in tqdm(range(args.trnsz)):
        syndromes[i] = imgsdr2ZX(image_syndromes[i])
    syndromes_new = f.create_dataset('syndromes', data=syndromes,
                                     chunks=True, compression="gzip")
    logical_errors_new = f.create_dataset('logical_errors', data=logical_errors, chunks=True, compression="gzip")
log("img2seq... Done.")
# python3 img2seq_sur.py --c_type sur --d 3 --p 0.1 --trnsz 10000000
# python3 img2seq_sur.py --c_type sur --d 5 --p 0.1 --trnsz 10000000
# nohup python3 img2seq_sur.py --c_type sur --d 7 --p 0.1 --trnsz 10000000 &
# nohup python3 img2seq_sur.py --c_type sur --d 9 --p 0.1 --trnsz 10000000 &
# nohup python3 img2seq_sur.py --c_type sur --d 11 --p 0.1 --trnsz 10000000 &
# nohup python3 img2seq_sur.py --c_type sur --d 13 --p 0.1 --trnsz 10000000 &

