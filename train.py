from args import args
from network import get_model
from data_process import get_loader
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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, global_step):
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


wandb_name = f"d{args.d}_p{args.p}_trnsz{args.trnsz}_ep{args.epoch}"
if args.zip == 1:
    wandb_name += f"_zip_limit{args.limit}"
wandb.init(
    project="ffn",
    name=wandb_name,
    config={
        'd': args.d,
        'p': args.p,
        'train size': args.trnsz,
        'epoch': args.epoch,
        'hidden size': config.hidden_size,
        'hidden layer': config.hidden_layer,
        'learning rate': config.learning_rate
    }
)

# device = torch.device('cpu')
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

model = get_model()
model.to(device)
trainloader, testloader = get_loader()
set_seed(args)
criterion = nn.BCEWithLogitsLoss()
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


        # Forward pass
        outputs = model(x)
        # log("y (shape={}): \n{}".format(y.shape, y))
        # log("y.type: {}".format(y.dtype))
        # log("outputs (shape={}): \n{}".format(outputs.shape, outputs))
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'train loss': loss.item()})
        epoch_iterator.set_description("Training (%d / %d Steps)(loss=%2.5f)" % (i + 1, total_step, loss.item()))

        if (i + 1) % args.eval_every == 0:
            accuracy = valid(args, model, testloader, i + 1)
            if best_acc < accuracy:
                best_acc = accuracy
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
