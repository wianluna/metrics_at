import numpy as np
import torch
import lpips
import argparse
import os
from common.data.bapps.data_loader import CreateDataLoader
from common.attacks import NoAttack
from common import attacks
import tqdm

def train(model, attack, num_epochs, num_epochs_decay, train_dataloader):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    rankloss = lpips.BCERankingLoss()
    model.to(device)
    rankloss.to(device)
    model.train()
    rankloss.train()
    lr = 0.001
    optimizer = torch.optim.Adam(list(model.parameters()) + list(rankloss.parameters()), lr=lr)
    for ep in range(num_epochs + num_epochs_decay):
        correct = 0
        d0s = []
        d1s = []
        gts = []
        for data in tqdm.tqdm(train_dataloader.load_data()):
            ref = data['ref'].to(device=device)
            p0 = data['p0'].to(device=device)
            p1 = data['p1'].to(device=device)
            input_judge = data['judge'].to(device=device)
            attacked_p0, attacked_p1 = attack.attack_pairs(model, ref, p0, p1, input_judge)
            d0 = model.forward(ref, attacked_p0)
            d1 = model.forward(ref, attacked_p1)
            d0s += d0.cpu().detach().numpy().flatten().tolist()
            d1s += d1.cpu().detach().numpy().flatten().tolist()
            gts += input_judge.cpu().detach().numpy().flatten().tolist()
            optimizer.zero_grad()
            loss = rankloss.forward(d0, d1, input_judge * 2. - 1.)
            loss.sum().backward()
            optimizer.step()

        d0s = np.array(d0s)
        d1s = np.array(d1s)
        gts = np.array(gts)
        outputs = np.stack((d1s, d0s), axis=1)
        correct = outputs.argmax(1) == np.round(gts)
        scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
        print(f"train scores: {np.mean(correct)}, {np.mean(scores)})")
        if ep >= opt.nepoch:
            lr -= lr / opt.nepoch_decay
            print(f"new lr: {lr}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        #torch.save(model.state_dict(), model_save_path)

parser = argparse.ArgumentParser()

parser.add_argument('--train_datasets', type=str, nargs='+', default=['train/traditional','train/cnn','train/mix'], help='datasets to train on')
parser.add_argument('--dataroot', type=str, default=".", help='dataset root')

parser.add_argument('--save_dir', type=str, default="./models", help='directory to save models in')
parser.add_argument('--train_attacks', type=str, default=["FGSM", "PGD"], help='attack types for training')
parser.add_argument('--train_epses', type=float, nargs='+', default=[2, 4, 8, 10], help='attack epsilons for training')

parser.add_argument('--train_dummy', action='store_true', help="evaluate default lpips without adversarial training")
parser.add_argument('--test_dummy', action='store_true', help="test model response without attacks")

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
parser.add_argument('--pretrained', action='store_true', help="use pretrained model")

parser.add_argument('--nepoch', type=int, default=1, help='# epochs at base learning rate')
parser.add_argument('--nepoch_decay', type=int, default=1, help='# additional epochs at linearly learning rate')


opt = parser.parse_args()


train_dataloader = CreateDataLoader(opt.train_datasets, data_root=opt.dataroot, batch_size=opt.batch_size, serial_batches=False, nThreads=opt.num_workers)

train_attacks = []

if opt.train_dummy:
    train_attacks.append(NoAttack())

for train_attack_cls in opt.train_attacks:
    for train_eps in opt.train_epses:
        attack_cls = getattr(attacks, train_attack_cls)
        attack = attack_cls(eps=train_eps / 255, proba=0.5)
        train_attacks.append(attack)

for train_attack in train_attacks:
    print(f"training model {train_attack.get_name()}")
    model = lpips.LPIPS(pretrained=opt.pretrained, pnet_rand=not opt.pretrained)
    train(model, train_attack, opt.nepoch, opt.nepoch_decay, train_dataloader)
    model_save_path = os.path.join(opt.save_dir, train_attack.get_name() + ".pt")
    torch.save(model.state_dict(), model_save_path)
