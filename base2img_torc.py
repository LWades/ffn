from args import args
from utils import log, ZX2image
import h5py

path_data = '/root/Surface_code_and_Toric_code/{}_pe/'.format(args.c_type)
filename_read_data = '{}_d{}_p{}_trnsz{}_seed0.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), args.poolsz)
filename_write_data = '{}_d{}_p{}_trnsz{}_imgsdr_seed0.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), args.trnsz)

log("base to img...")
with h5py.File(path_data + filename_read_data, 'r') as f:
    syndromes = f['syndromes'][()]
    num = syndromes.shape[0]
    logical_errors = f['logical_errors'][()]
    image_syndromes = ZX2image(args.d, syndromes)
with h5py.File(path_data + filename_write_data, 'w') as f:
    syndromes_cutoff = f.create_dataset('image_syndromes', data=syndromes[num],
                                     chunks=True, compression="gzip")
    logical_errors_cutoff = f.create_dataset('logical_errors', data=logical_errors[num], chunks=True, compression="gzip")
log("base to img... Done.")

# python3 base2img_torc.py --c_type torc --d 3 --p 0.01 --trnsz 10000000
