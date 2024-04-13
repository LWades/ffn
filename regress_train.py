from args import args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler, SequentialSampler
import numpy as np
import torch.nn.functional as F  # 添加这行导入 F
import wandb
from utils import log
import h5py
import random
from tqdm import tqdm
import os

# 假设 d 为图像的宽度和高度
d = 128
in_channels = 1  # 如果是三通道图像
out_channels = 1  # 假设输出也是三通道
file_name_train_x = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imger_seed{}.hdf5".format(args.c_type, args.c_type, args.d, format(args.p, '.3f'), args.trnsz, args.seed)
file_name_test_x = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imger_eval_seed{}.hdf5".format(args.c_type, args.c_type, args.d, format(args.p, '.3f'), 10000, args.eval_seed)
file_name_train_y = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imgsdr_seed{}.hdf5".format(args.c_type, args.c_type, args.d, format(args.p, '.3f'), args.trnsz, args.seed)
file_name_test_y = "/root/Surface_code_and_Toric_code/{}_pe/{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}.hdf5".format(args.c_type, args.c_type, args.d, format(args.p, '.3f'), 10000, args.eval_seed)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SurImgDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['image_syndromes'])

    def __getitem__(self, idx):
        syndrome_data = self.data['image_syndromes'][idx]
        image_error_data = self.data['image_errors'][idx]
        # log("1 {}".format(syndrome_data))
        # log("2 {}".format(logical_error_data))
        return syndrome_data, image_error_data


def save_model(model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    log("Saved model checkpoint to [DIR: {}]".format(args.output_dir))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


# 定义模型
class RegressionCNN(nn.Module):
    def __init__(self, in_channels, out_channels, image_size):
        super(RegressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x


def get_loader_sur_img():
    log("Data loading...")
    # traindata = np.load(path_data + filename_train_data)
    # traindata = np.load('/root/qecGPT/qec/trndt/sur_d3_p0.010_trnsz10000000_imgsdr.npz')
    # testdata = np.load('/root/qecGPT/qec/trndt/sur_d3_p0.010_trnsz10000_imgsdr_eval.npz')
    # testdata = np.load(path_data + filename_test_data)
    log("traindata: {}".format(file_name_train_x))
    log("traindata: {}".format(file_name_train_y))
    with h5py.File(file_name_train_x, 'r') as f, h5py.File(file_name_train_y, 'r') as f2:
        train_syndrome = f['image_errors'][()]
        log("traindata size: {}".format(train_syndrome.shape[0]))
        train_syndrome_post = np.expand_dims(train_syndrome, axis=1)
        train_target_error = f2['image_syndromes'][()]
        trainset = SurImgDataset({'image_errors': train_syndrome_post, 'image_syndromes': train_target_error})
    log("testdata: {}".format(file_name_test_x))
    log("testdata: {}".format(file_name_test_y))
    with h5py.File(file_name_test_x, 'r') as f, h5py.File(file_name_test_y, 'r') as f2:
        test_syndrome = f['image_errors'][()]
        log("testdata size: {}".format(test_syndrome.shape[0]))
        test_syndrome_post = np.expand_dims(test_syndrome, axis=1)
        test_target_error = f2['image_syndromes'][()]
        testset = SurImgDataset({'image_errors': test_syndrome_post, 'image_syndromes': test_target_error})
    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             pin_memory=True) if testset is not None else None
    log("Data loading... Done.")
    return train_loader, test_loader


model = RegressionCNN(in_channels, out_channels, d)

# 损失函数和优化器
loss_function = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据加载器（这里需要用实际数据替换 np.random.randn 示例）
# train_data = TensorDataset(torch.Tensor(np.random.randn(100, in_channels, d, d)),
#                            torch.Tensor(np.random.randn(100, out_channels, d, d)))
# train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
train_loader, test_loader = get_loader_sur_img()


# 训练循环
def train_epoch(model, train_loader, loss_function, optimizer):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


# 验证循环
def validate_epoch(model, val_loader, loss_function):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def valid_1d(model, test_loader):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    # targets = torch.tensor([0, 1, 2, 3])
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        # x = x.to(torch.float16)
        # y = y.to(torch.long)
        # y = torch.unsqueeze(y, 1)
        y = y.to(torch.float)
        with torch.no_grad():
            x = x.unsqueeze(1)
            logits = model(x)
            # logits = model(x)[0]
            log(f"logits shape: {logits.shape}")
            log(f"y shape: {y.shape}")

            eval_loss = loss_function(logits, y)
            eval_losses.update(eval_loss.item())
            # errors = torch.abs(logits - targets) / torch.abs(targets)
            errors = torch.abs(logits - y) / torch.abs(y)
            confidence_threshold = 0.2
            preds = torch.zeros_like(logits)
            preds[errors < confidence_threshold] = logits[errors < confidence_threshold]
            preds = torch.round(preds)
            log("preds: \n{}".format(preds))
            # log()
            # preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = (all_preds == all_label).mean()

    log("Valid Loss: %2.5f" % eval_losses.avg)
    log("Valid Accuracy: %2.5f" % accuracy)
    wandb.log({'valid_loss': eval_losses.avg, 'valid_accuracy': accuracy})

    return accuracy


# 示例训练和验证过程
# num_epochs = args.epoch
# for epoch in range(num_epochs):
#     train_loss = train_epoch(model, train_loader, loss_function, optimizer)
#     # val_loss = validate_epoch(model, val_loader, loss_function)  # 需要定义 val_loader
#     print(f'Epoch {epoch + 1}, Train Loss: {train_loss}')  # , Validation Loss: {val_loss}

wandb_name = f"rg_{args.nn}_d{args.d}_p{args.p}_trnsz{args.trnsz}_ep{args.epoch}"
wandb_project = "work02"
wandb.init(
    project=wandb_project,
    name=wandb_name,
    config={
        'd': args.d,
        'p': args.p,
        'train size': args.trnsz,
        'epoch': args.epoch,
    }
)


if __name__ == "__main__":
    # device = torch.device('cpu')
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model.to(device)

    set_seed(args)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_step = len(train_loader)
    best_acc = 0
    log("Training Start...")
    log("Epoch num: ".format(args.epoch))
    for epoch in range(args.epoch):
        log("Epoch: {}".format(epoch + 1))
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True
                              )
        for i, batch in enumerate(epoch_iterator):
            # Move tensors to the configured device
            batch = tuple(t.to(device) for t in batch)
            x, y = batch

            if args.nn == 'cnn':
                y = y.to(torch.float)
                y = y.squeeze()

            x = x.unsqueeze(1)
            # Forward pass
            outputs = model(x)
            # log("outputs.type: {}".format(outputs.dtype))
            # log("y (shape={}): \n{}".format(y.shape, y))
            # log("y.type: {}".format(y.dtype))
            # log("outputs (shape={}): \n{}".format(outputs.shape, outputs))
            loss = loss_function(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item()})
            epoch_iterator.set_description("Training (%d / %d Steps)(loss=%2.5f)" % (i + 1, total_step, loss.item()))

            if (i + 1) % args.eval_every == 0:
                if args.nn == 'cnn':
                    accuracy = valid_1d(model, test_loader)
                else:
                    log("Error nn")
                    exit(1)
                if best_acc < accuracy:
                    best_acc = accuracy
                    save_model(model)
                    wandb.log({'Best Accuracy until Now': best_acc})
                model.train()
    log("Best Accuracy: {}".format(best_acc))
    log("Training... Done!")

# nohup python3 regress_train.py --name cnn_rg_11_0.10_1e7 --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 1 --work 2 > logs/cnn_rg_11_0.10_1e7.log &
