
from utils import log
import torch
import h5py
from args import args
from network import get_model

key_syndrome = 'image_syndromes'
key_logical_error = 'logical_errors'
pwd_trndt = '/root/Surface_code_and_Toric_code/{}_pe/'.format(args.c_type)
pwd_model = '/root/ffn/output/'
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

args.sym = 'all'
model = get_model()
p = 0.1
i = 0

log("Eval...")

s = ""
for s_type in args.sym:
    s = s + "_" + s_type

filename_test_data_base = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}_base_{}.hdf5'.format(args.c_type, args.d, format(p, '.3f'), args.trnsz, args.eval_seed, s)
filename_test_data_sym = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}_sym_{}.hdf5'.format(args.c_type, args.d, format(p, '.3f'), args.trnsz, args.eval_seed, s)
log("test_data_base: {}".format(filename_test_data_base))
log("test_data_sym: {}".format(filename_test_data_sym))
with h5py.File(filename_test_data_base, 'r') as f_b, h5py.File(filename_test_data_sym, 'r') as f_s:
    test_syndrome_base = f_b[key_syndrome][()]
    test_logical_error_base = f_b[key_logical_error][()]
    test_syndrome_sym = f_s[key_syndrome][()]
    test_logical_error_sym = f_s[key_logical_error][()]
    # testset_base = SurDataset({key_syndrome: test_syndrome_base, key_logical_error: test_logical_error_base})
    # testset_sym = SurDataset({key_syndrome: test_syndrome_sym, key_logical_error: test_logical_error_sym})

# test_sampler_base = SequentialSampler(testset)
# test_loader_ = DataLoader(testset,
#                          sampler=test_sampler,
#                          batch_size=args.eval_batch_size,
#                          num_workers=2,
#                          pin_memory=True) if testset is not None else None

# log("test_loader.dataset type: {}".format(type(test_loader.dataset)))
# log("test_loader.dataset: {}".format(test_loader.dataset))

    model_name = '{}_checkpoint.bin'.format(args.name)
    # model_name = 'fnn_{}_{}-5e6_checkpoint.bin'.format(args.d, format(args.p, '.2f'))
    log("model: {}".format(model_name))
    state_dict = torch.load(pwd_model + model_name)
    state_dict = {k: v for k, v in state_dict.items() if
                  k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(torch.load(pwd_model + model_name))
    model.to(device)
    model.eval()

    c_eq_true, c_eq_false, c_neq = 0, 0, 0
    for i in range(10000):
        y_base = test_logical_error_base[i]
        logit_base = model(test_syndrome_base[i])
        pred_base = torch.argmax(logit_base)

        y_sym = test_logical_error_sym[i]
        logit_sym = model(test_syndrome_sym[i])
        pred_sym = torch.argmax(logit_sym)

        result_base = (pred_base == y_base)
        result_sym = (pred_sym == y_sym)

        if result_base == result_sym:
            if result_base:
                c_eq_true += 1
            else:
                c_eq_false += 1
        else:
            c_neq += 1
        log("c_eq_true: {}, c_eq_false:{}, c_neq: {}").format(c_eq_true, c_eq_false, c_neq, end='\r')
    log("finish: c_eq_true: {}, c_eq_false:{}, c_neq: {}").format(c_eq_true, c_eq_false, c_neq)
    # i += 1
    # log("p {} acc: {}".format(format(p, '.3f'), acc))
    # log("accs: \n{}".format(accs))
    log("Eval... Done.")
# nohup python3 eval_vision.py --name cnn_11_0.10-5e6-o --nn cnn --c_type sur --d 11 --k 1 --p 0.10 --eval_seed 3 --trnsz 10000 --gpu 1 --sym all > logs/ef3.log &
