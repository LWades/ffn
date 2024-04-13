from args import args
from network import get_model
from data_process import get_loader, get_loader_sur_img
import config as configs
from utils import log
import wandb
import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

CONFIGS = {
    'ffn': configs.get_ffn_torc(),
}
config = CONFIGS['ffn']


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


def valid_1d(model, test_loader):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        # x = x.to(torch.float16)
        # y = y.to(torch.long)
        y = torch.squeeze(y)
        y = y.to(torch.long)
        if args.nn == 'rnn':
            x = x.unsqueeze(1)
            x = x.float()
        with torch.no_grad():
            logits = model(x)
            # logits = model(x)[0]
            # log(f"logits shape: {logits.shape}")
            # log(f"y shape: {y.shape}")

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

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


def valid_1d_eval(model, test_loader):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        # x = x.to(torch.float16)
        # y = y.to(torch.long)
        y = torch.squeeze(y)
        x = torch.unsqueeze(x, 1)
        log("x shape: {}".format(x.shape))
        # x = torch.squeeze(x)
        # x = x.permute(1, 0, 2, 3)
        y = y.to(torch.long)
        with torch.no_grad():
            # x  = x.permute(1,0,2,3)
            logits = model(x)
            # logits = model(x)[0]
            # log(f"logits shape: {logits.shape}")
            # log(f"y shape: {y.shape}")

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

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

    return accuracy

def valid(model, test_loader):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          )
    loss_fct = nn.BCEWithLogitsLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            # log("logits (shape={}): {}".format(logits.shape, logits))
            probs = torch.sigmoid(logits)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = (probs > 0.5).long()
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

    # log("Validation Results")
    log("Valid Loss: %2.5f" % eval_losses.avg)
    log("Valid Accuracy: %2.5f" % accuracy)
    wandb.log({'valid_loss': eval_losses.avg, 'valid_accuracy': accuracy})

    return accuracy


def valid_eval(model, test_loader):
    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          )
    loss_fct = nn.BCEWithLogitsLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)
            # log("logits (shape={}): {}".format(logits.shape, logits))
            probs = torch.sigmoid(logits)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = (probs > 0.5).long()
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

    # log("Validation Results")
    log("Valid Loss: %2.5f" % eval_losses.avg)
    log("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # device = torch.device('cpu')

    model = get_model()
    model.to(device)
    if args.nn == 'cnn':
        trainloader, testloader = get_loader_sur_img()
    elif args.nn == 'fnn':
        trainloader, testloader = get_loader()
    elif args.nn == 'rnn':
        trainloader, testloader = get_loader()

    wandb_name = f"{args.nn}_d{args.d}_p{args.p}_trnsz{args.trnsz}_ep{args.epoch}"
    if args.zip == 1:
        wandb_name += f"_zip_limit{args.limit}"
    if args.work == 1:
        wandb_project = "work01"
    else:
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

    set_seed(args)
    if args.nn == 'fnn':
        criterion = nn.BCEWithLogitsLoss()
    elif args.nn == 'cnn' or args.nn == 'rnn':
        criterion = nn.CrossEntropyLoss()
    else:
        log("?")
        exit(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_step = len(trainloader)
    best_acc = 0
    log("Training Start...")
    log("Epoch num: ".format(args.epoch))
    for epoch in range(args.epoch):
        log("Epoch: {}".format(epoch + 1))
        model.train()
        epoch_iterator = tqdm(trainloader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True
                              )
        for i, batch in enumerate(epoch_iterator):
            # Move tensors to the configured device
            batch = tuple(t.to(device) for t in batch)
            x, y = batch

            log("x.shape: {}".format(x.shape))
            if args.nn == 'cnn':
                y = y.to(torch.long)
                y = y.squeeze()
            elif args.nn == 'rnn':
                x = x.unsqueeze(1)
                x = x.float()
                y = y.to(torch.long)
                y = y.squeeze()
            # x.unsqueeze(0)
            # log("x.shape: {}".format(x.shape))
            # Forward pass
            outputs = model(x)
            # log("outputs.type: {}".format(outputs.dtype))
            # log("y (shape={}): \n{}".format(y.shape, y))
            # log("y.type: {}".format(y.dtype))
            # log("outputs (shape={}): \n{}".format(outputs.shape, outputs))
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item()})
            epoch_iterator.set_description("Training (%d / %d Steps)(loss=%2.5f)" % (i + 1, total_step, loss.item()))

            if (i + 1) % args.eval_every == 0:
                if args.nn == 'fnn':
                    accuracy = valid(model, testloader)
                elif args.nn == 'cnn' or args.nn == 'rnn':
                    accuracy = valid_1d(model, testloader)
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
# python3 train.py --c_type torc --d 3 --k 2 --p 0.010 --epoch 10
# nohup python3 train.py --c_type torc --d 3 --k 2 --p 0.010 --epoch 10 > logs/ffn_d3_k2_p0.010_e10.log &
# nohup python3 train.py --c_type torc --d 3 --k 2 --p 0.030 --epoch 10 > logs/ffn_d3_k2_p0.020_e10.log &
# nohup python3 train.py --c_type torc --d 3 --k 2 --p 0.030 --epoch 5 > logs/ffn_d3_k2_p0.030_e5.log &
# nohup python3 train.py --c_type torc --d 3 --k 2 --p 0.040 --epoch 10 > logs/ffn_d3_k2_p0.040_e10_3.log &
# nohup python3 train.py --c_type torc --zip 1 --limit 3000 --d 5 --k 2 --p 0.040 --epoch 10 --trnsz 5000000 --gpu 1 --sym tl > logs/ffn_zip1_d5_k2_p0.040_e10.log &
# nohup python3 train.py --c_type torc --zip 1 --limit 4000 --d 5 --k 2 --p 0.040 --epoch 10 --trnsz 5000000 --gpu 0 --sym tl > logs/ffn_zip1_d5_k2_p0.040_e10_2.log &

# nohup python3 train.py --c_type torc --d 5 --k 2 --p 0.040 --epoch 20 --trnsz 5000000 > logs/ffn_d5_k2_p0.040_e20_0.log &
# nohup python3 train.py --c_type torc --zip 1 --limit 3000 --d 5 --k 2 --p 0.040 --epoch 20 --trnsz 5000000 --gpu 1 --sym tl > logs/ffn_d5_k2_p0.0sf40_e20sf_0.log &


# nohup python3 train.py --nn cnn --c_type sur --d 7 --k 1 --p 0.010 --epoch 20 --trnsz 10000000 > logs/cnn_d7_k1_p0.010_e20_0.log &
# python3 train.py --nn cnn --c_type sur --d 3 --k 1 --p 0.010 --epoch 20 --trnsz 10000000
# python3 train.py --nn cnn --c_type sur --d 5 --k 1 --p 0.010 --epoch 20 --trnsz 10000000
# python3 train.py --nn cnn --c_type sur --d 11 --k 1 --p 0.010 --epoch 20 --trnsz 10000000
# nohup python3 train.py --nn cnn --c_type sur --d 5 --k 1 --p 0.010 --epoch 20 --trnsz 10000000 > logs/cnn_d5_k1_p0.010_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 11 --k 1 --p 0.010 --epoch 20 --trnsz 10000000 --gpu 0 > logs/cnn_d11_k1_p0.010_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 11 --k 1 --p 0.050 --epoch 20 --trnsz 10000000 --gpu 2 > logs/cnn_d11_k1_p0.050_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 2 > logs/cnn_d11_k1_p0.100_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 11 --k 1 --p 0.150 --epoch 20 --trnsz 10000000 --gpu 2 > logs/cnn_d11_k1_p0.150_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 7 --k 1 --p 0.010 --epoch 20 --trnsz 10000000 --gpu 2 > logs/cnn_d7_k1_p0.010_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 7 --k 1 --p 0.050 --epoch 20 --trnsz 10000000 --gpu 3 > logs/cnn_d7_k1_p0.050_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 7 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 2 > logs/cnn_d7_k1_p0.100_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 7 --k 1 --p 0.150 --epoch 20 --trnsz 10000000 --gpu 0 > logs/cnn_d7_k1_p0.150_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 7 --k 1 --p 0.200 --epoch 20 --trnsz 10000000 --gpu 3 > logs/cnn_d7_k1_p0.200_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 3 --k 1 --p 0.010 --epoch 20 --trnsz 10000000 --gpu 2 > logs/cnn_d3_k1_p0.010_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 3 --k 1 --p 0.050 --epoch 20 --trnsz 10000000 --gpu 3 > logs/cnn_d3_k1_p0.050_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 3 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 2 > logs/cnn_d3_k1_p0.100_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 3 --k 1 --p 0.150 --epoch 20 --trnsz 10000000 --gpu 0 > logs/cnn_d3_k1_p0.150_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 3 --k 1 --p 0.200 --epoch 20 --trnsz 10000000 --gpu 3 > logs/cnn_d3_k1_p0.200_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 9 --k 1 --p 0.010 --epoch 20 --trnsz 10000000 --gpu 0 > logs/cnn_d9_k1_p0.200_e20_0.log &

# torc 5
# nohup python3 train.py --nn cnn --c_type sur --d 5 --k 1 --p 0.010 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_d5_k1_p0.010_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 5 --k 1 --p 0.050 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_d5_k1_p0.050_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 5 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_d5_k1_p0.100_e20_0.log &
# nohup python3 train.py --nn cnn --c_type sur --d 5 --k 1 --p 0.150 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_d5_k1_p0.150_e20_0.log &
# nohup python3 train.py --name cnn_11_0.10-p-5e6 --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 5000000 --gpu 1 --work 2 > logs/cnn_d11_k1_p0.100_e20_7.log &
# nohup python3 train.py --name cnn_11_0.10-5e6 --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_d11_k1_p0.100_e20_2.log &
# nohup python3 train.py --name cnn_11_0.10-5e6 --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 5000000 --gpu 0 --work 2 > logs/cnn_d11_k1_p0.100_e20_4.log &
# nohup python3 train.py --name cnn_11_0.10-5e6-0409 --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 5000000 --gpu 0 --work 2 > logs/cnn_d11_k1_p0.100_e20_5.log &


# nohup python3 train.py --name cnn_5_0.10-1e7-p --nn cnn --c_type sur --d 5 --k 1 --p 0.10 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_5_0.10-1e7_0.log &

# fnn 3 5 7 5e6
# nohup python3 train.py --name fnn_3_0.10-5e6 --nn fnn --c_type torc --d 3 --k 2 --p 0.10 --epoch 20 --trnsz 5000000 --gpu 1 --work 1 > logs/fnn_3_0.10-5e6_0.log &
# nohup python3 train.py --name fnn_5_0.10-5e6 --nn fnn --c_type torc --d 5 --k 2 --p 0.10 --epoch 20 --trnsz 5000000 --gpu 2 --work 1 > logs/fnn_5_0.10-5e6_0.log &
# nohup python3 train.py --name fnn_7_0.10-5e6 --nn fnn --c_type torc --d 7 --k 2 --p 0.10 --epoch 20 --trnsz 5000000 --gpu 3 --work 1 > logs/fnn_7_0.10-5e6_0.log &
# nohup python3 train.py --name fnn_9_0.10-5e6 --nn fnn --c_type torc --d 9 --k 2 --p 0.10 --epoch 20 --trnsz 5000000 --gpu 0 --work 1 > logs/fnn_9_0.10-5e6_0.log &
# nohup python3 train.py --name fnn_3_0.07-5e6 --nn fnn --c_type torc --d 3 --k 2 --p 0.07 --epoch 20 --trnsz 5000000 --gpu 1 --work 1 > logs/fnn_3_0.07-5e6_0.log &
# nohup python3 train.py --name fnn_5_0.07-5e6 --nn fnn --c_type torc --d 5 --k 2 --p 0.07 --epoch 20 --trnsz 5000000 --gpu 2 --work 1 > logs/fnn_5_0.07-5e6_0.log &
# nohup python3 train.py --name fnn_7_0.07-5e6 --nn fnn --c_type torc --d 7 --k 2 --p 0.07 --epoch 20 --trnsz 5000000 --gpu 3 --work 1 > logs/fnn_7_0.07-5e6_0.log &
# nohup python3 train.py --name fnn_9_0.07-5e6 --nn fnn --c_type torc --d 9 --k 2 --p 0.07 --epoch 20 --trnsz 5000000 --gpu 0 --work 1 > logs/fnn_9_0.07-5e6_0.log &

# nohup python3 train.py --name fnn_7_0.07-5e6-zip-lm10000 --nn fnn --zip 1 --c_type torc --d 7 --k 2 --p 0.07 --sym all --limit 10000 --epoch 20 --trnsz 5000000 --gpu 0 --work 1 > logs/fnn_7_0.07-5e6-zip-lm10000.log &
# nohup python3 train.py --name fnn_3_0.07-5e6-zip-lm10000 --nn fnn --zip 1 --c_type torc --d 3 --k 2 --p 0.07 --sym all --limit 10000 --epoch 20 --trnsz 5000000 --gpu 2 --work 1 > logs/fnn_3_0.07-5e6-zip-lm10000.log &
# nohup python3 train.py --name fnn_5_0.07-5e6-zip-lm10000 --nn fnn --zip 1 --c_type torc --d 5 --k 2 --p 0.07 --sym all --limit 10000 --epoch 20 --trnsz 5000000 --gpu 2 --work 1 > logs/fnn_5_0.07-5e6-zip-lm10000.log &


# nohup python3 train.py --name cnn_11_0.10-5e6-i --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 5000000 --gpu 1 --work 2 > logs/cnn_d11_k1_p0.100_e20_10.log &
# nohup python3 train.py --name fnn_3_0.07-5e6 --nn fnn --c_type torc --d 3 --k 2 --p 0.07 --epoch 20 --trnsz 5000000 --gpu 1 --work 1 > logs/fnn_3_0.07-5e6_0.log &

# nohup python3 train.py --name cnn_11_0.10-5e6-o --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 5000000 --gpu 1 --work 2 > logs/cnn_d11_k1_p0.100_e20_11.log &

# nohup python3 train.py --name cnn_11_0.10-5e6-o --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 5000000 --gpu 1 --work 2 > logs/cnn_d11_k1_p0.100_e20_11.log &
# nohup python3 train.py --name cnn_11_0.10-1e7-o --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 1 --work 2 > logs/cnn_11_0.10-1e7-o.log &
# nohup python3 train.py --name cnn_9_0.10-1e6 --nn cnn --c_type sur --d 9 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 2 --work 2 > logs/cnn_9_0.10-1e6.log &


# nohup python3 train.py --name fnn_9_0.05-5e6 --nn fnn --c_type torc --d 9 --k 2 --p 0.05 --epoch 20 --trnsz 5000000 --gpu 3 --work 1 > logs/fnn_9_0.05-5e6_0.log &
# nohup python3 train.py --name fnn_9_0.05-5e6-zip-lm2000 --nn fnn --zip 1 --c_type torc --d 9 --k 2 --p 0.05 --sym all --limit 2000 --epoch 20 --trnsz 5000000 --gpu 3 --work 1 > logs/fnn_9_0.05-5e6-zip-lm2000.log &

# cnn 1e7
# nohup python3 train.py --name cnn_3_0.10_1e7 --nn cnn --c_type sur --d 3 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_3_0.10-1e7.log &
# exit
# nohup python3 train.py --name cnn_5_0.10_1e7 --nn cnn --c_type sur --d 5 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 1 --work 2 > logs/cnn_5_0.10-1e7.log &
# nohup python3 train.py --name cnn_7_0.10_1e7 --nn cnn --c_type sur --d 7 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_7_0.10-1e7.log &
# ing
# nohup python3 train.py --name cnn_9_0.10_1e7 --nn cnn --c_type sur --d 9 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 2 --work 2 > logs/cnn_9_0.10-1e7.log &
# exit
# nohup python3 train.py --name cnn_11_0.10_1e7 --nn cnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/cnn_11_0.10-1e7.log &
# ing
# nohup python3 train.py --name cnn_13_0.10_1e7 --nn cnn --c_type sur --d 13 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 1 --work 2 > logs/cnn_13_0.10-1e7.log &
# ing

# rnn
# nohup python3 train.py --name rnn_3_0.10_1e7 --nn rnn --c_type sur --d 3 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/rnn_3_0.10-1e7.log &
# nohup python3 train.py --name rnn_7_0.10_1e7 --nn rnn --c_type sur --d 7 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 0 --work 2 > logs/rnn_7_0.10-1e7.log &
# nohup python3 train.py --name rnn_11_0.10_1e7 --nn rnn --c_type sur --d 11 --k 1 --p 0.100 --epoch 20 --trnsz 10000000 --gpu 2 --work 2 > logs/rnn_11_0.10-1e7.log &

# nohup python3 train.py --name fnn_9_0.05-5e6-zip-tl-lm50 --nn fnn --zip 1 --c_type torc --d 9 --k 2 --p 0.05 --sym tl --limit 50 --epoch 20 --trnsz 5000000 --gpu 0 --work 1 > logs/fnn_9_0.05-5e6-zip-tl-lm50.log &
