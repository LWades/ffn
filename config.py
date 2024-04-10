import ml_collections
from args import args
from utils import log


def get_ffn_torc():
    config = ml_collections.ConfigDict()
    config.input_size = 2 * args.d ** 2 - args.k
    config.hidden_size = 10
    config.hidden_layer = 2
    config.num_classes = 2 ** args.k
    config.batch_size = 512
    # config.learning_rate = 0.0001
    # config.learning_rate = 0.000001
    # config.learning_rate = 0.00001
    # config.learning_rate = 5e-4
    # config.learning_rate = 1e-3
    # config.learning_rate = 1e-4
    config.learning_rate = 0.00075  # 当前最佳
    # config.learning_rate = 0.00025
    config.num_epochs = 10
    return config


def get_cnn_sur_3():
    config = ml_collections.ConfigDict()
    config.DL = 1
    config.CL = 3
    config.N = 512
    return config


def get_cnn_sur_5():
    config = ml_collections.ConfigDict()
    config.DL = 1
    config.CL = 3
    config.N = 512
    return config


def get_cnn_sur_7():
    config = ml_collections.ConfigDict()
    config.DL = 1
    config.CL = 3
    config.N = 512
    return config


def get_cnn_sur_9():
    config = ml_collections.ConfigDict()
    config.DL = 1
    config.CL = 4
    config.N = 512
    return config


# 04092208
def get_cnn_sur_11():
    config = ml_collections.ConfigDict()
    config.DL = 1
    config.CL = 4
    config.N = 512
    return config

# def get_cnn_sur_11():
#     config = ml_collections.ConfigDict()
#     config.DL = 2
#     config.CL = 6
#     config.N = 1024
#     return config


def get_cnn_torc_3():
    pass


def get_cnn_torc_5():
    pass


def get_cnn_torc_7():
    pass


def get_cnn_torc_9():
    pass


def get_cnn_torc_11():
    pass

