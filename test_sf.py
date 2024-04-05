from args import args
from symmetric_syndrome import ZX2image, SymmetricSyndrome
from utils import log
import h5py
import wandb
from tqdm import tqdm
import numpy as np
import copy
from symmetric_filter import SymmetricFilter

s_types, d, rep_limit, trnsz = args.sym, args.d, args.limit, args.trnsz

symmetric_filter = SymmetricFilter(d, s_types, rep_limit)

syndrome01 = np.array([0, 1, 0,
                       0, 0, 1,
                       0, 0, 0,
                       0, 0, 1,
                       0, 0, 1,
                       0, 0, 0])
syndrome01 = syndrome01[1:-1]
log(syndrome01)
# syndrome02 = np.array([0, 0, 0,
#                        1, 0, 0,
#                        0, 1, 0,
#                        0, 0, 0,
#                        0, 0, 0,
#                        1, 1, 0])
# rf:3
syndrome04 = np.array([0, 0, 0,
                       1, 1, 0,
                       0, 0, 0,
                       0, 0, 0,
                       1, 0, 0,
                       0, 1, 0])
# rt:0
# syndrome02 = np.array([0, 0, 0,
#                        1, 1, 0,
#                        0, 0, 0,
#                        0, 0, 0,
#                        0, 1, 0,
#                        1, 0, 0])
# rt:1
syndrome02 = np.array([0, 0, 0,
                       1, 0, 0,
                       1, 0, 0,
                       0, 0, 0,
                       1, 0, 0,
                       0, 1, 0])
# rt:2
syndrome03 = np.array([0, 0, 0,
                       1, 1, 0,
                       0, 0, 0,
                       0, 0, 0,
                       0, 0, 1,
                       0, 1, 0])
# syndrome02 = np.array([0, 1, 1,
#                        0, 0, 0,
#                        0, 0, 0,
#                        0, 0, 1,
#                        0, 1, 0,
#                        0, 0, 0])
syndrome02 = syndrome02[1:-1]
syndrome03 = syndrome03[1:-1]
syndrome04 = syndrome04[1:-1]
log(syndrome02)
f1 = symmetric_filter.filter(syndrome01)

log(f1)
log(symmetric_filter.get_size())

f2 = symmetric_filter.filter(syndrome02)

log(f2)
log(symmetric_filter.get_size())

f3 = symmetric_filter.filter(syndrome03)

log(f3)
log(symmetric_filter.get_size())

f4 = symmetric_filter.filter(syndrome04)

log(f4)
log(symmetric_filter.get_size())
