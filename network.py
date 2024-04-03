import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import config as configs
from args import args
from utils import log


CONFIGS = {
    'ffn': configs.get_ffn_torc(),
    'cnn_sur_3': configs.get_cnn_sur_3(),
    'cnn_sur_5': configs.get_cnn_sur_5(),
    'cnn_sur_7': configs.get_cnn_sur_7(),
    'cnn_sur_11': configs.get_cnn_sur_11()
}

if args.nn == 'fnn':
    config = CONFIGS['ffn']
elif args.nn == 'cnn':
    if args.d == 11:
        config = CONFIGS['cnn_sur_11']
    elif args.d == 7:
        config = CONFIGS['cnn_sur_7']
    elif args.d == 5:
        config = CONFIGS['cnn_sur_5']
    elif args.d == 3:
        config = CONFIGS['cnn_sur_3']

input_size = CONFIGS['ffn'].input_size
hidden_size = CONFIGS['ffn'].hidden_size
num_classes = CONFIGS['ffn'].num_classes
num_epochs = CONFIGS['ffn'].num_epochs
batch_size = CONFIGS['ffn'].batch_size
learning_rate = CONFIGS['ffn'].learning_rate


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.hid = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        for i in range(config.hidden_layer):
            out = self.hid(out)
            out = self.relu(out)
        out = self.fc2(out)
        return out


class CNN(nn.Module):
    def __init__(self, CL, DL, N, output_size):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(CL):
            in_channels = 1 if i == 0 else 32 * (2 ** (i - 1))
            out_channels = 32 * (2 ** i)
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.BatchNorm2d(out_channels))

            if args.d > 3:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.flatten = nn.Flatten()
        self.dense_layers = nn.ModuleList()

        # 计算卷积层输出的维度以连接到全连接层
        # self.conv_output_size = out_channels * (64 // 2 ** CL) * (64 // 2 ** CL)
        # out_channels = 32 * (2 ** (CL - 1))
        # feature_map_size = 64 // (2 ** (CL // 2))
        # self.conv_output_size = out_channels * feature_map_size * feature_map_size
        if args.d == 3:
            feature_map_size = 5
            self.conv_output_size = out_channels * feature_map_size * feature_map_size
        else:
            feature_map_size = 2 * args.d - 1
            self.conv_output_size = out_channels * (feature_map_size // 2 ** CL) * (feature_map_size // 2 ** CL)

        for i in range(DL):
            in_features = self.conv_output_size if i == 0 else N
            self.dense_layers.append(nn.Linear(in_features, N))
            self.dense_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(N, output_size)


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.dense_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 输出层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # RNN 前向传播
        out, _ = self.rnn(x, h0)
        # 获取最后时刻的输出并传给线性层
        out = self.linear(out[:, -1, :])
        return out

# # 参数设定
# input_size = 10  # 输入大小
# hidden_size = 20  # 隐藏层大小
# output_size = 5  # 输出大小
#
# # 实例化模型
# model = SimpleRNN(input_size, hidden_size, output_size)
# print(model)


def get_model():
    if args.nn == 'fnn':
        model = FNN(input_size, hidden_size, num_classes)
    elif args.nn == 'cnn':
        log("init cnn...")
        model = CNN(config.CL, config.DL, config.N, 4)
        log("init cnn... Done.")
    return model


