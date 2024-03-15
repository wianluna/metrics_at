import numpy as np
import torch
import lpips
import argparse
from tqdm import tqdm
import os
from common.data.kadid10k.kadid10k import CustomImageDataset
from common.attacks import NoAttack
from common import attacks
from common.score_2afc_dataset import score_2afc_dataset
from scipy import stats

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', type=str, default=".", help='dataset root')

parser.add_argument('--save_dir', type=str, default="./models", help='directory where the trained models are')
parser.add_argument('--experiment_dir', type=str, default="./kadid_stats", help='path to save experiment results')

parser.add_argument('--train_attacks', type=str, default=[], help='attack types used for training')
parser.add_argument('--train_epses', type=str, default=[2, 4, 8, 10], help='attack epsilons used for training')

parser.add_argument('--val_attacks', type=str, default=["FGSM", "FreeFGSM", "PGD"], help='attack types for validation')
parser.add_argument('--val_epses', type=str, default=[2, 4, 6, 8, 10], help='attack epsilons for validation')

parser.add_argument('--train_dummy', action='store_true', help="evaluate model which wasn't adversarily trained")
parser.add_argument('--test_dummy', action='store_true', help="validate model response without attacks")
parser.add_argument('--test_r_lpips', action='store_true', help="validate R-LPIPS")

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

val_dataloader = torch.utils.data.DataLoader(CustomImageDataset(opt.dataroot), batch_size=opt.batch_size, num_workers = opt.num_workers)

trained_models_names = []
val_attacks = []

if opt.train_dummy:
    trained_models_names.append("NoAttack")

if opt.test_r_lpips:
    trained_models_names.append("latest_net_linf_x0")

if opt.test_dummy:
    val_attacks.append(NoAttack())

for train_attack_cls in opt.train_attacks:
    for train_eps in opt.train_epses:
        attack_cls = getattr(attacks, train_attack_cls)
        attack = attack_cls(eps=train_eps / 255, proba=0.5)
        trained_models_names.append(attack.get_name())

for val_attack_cls in opt.val_attacks:
    for val_eps in opt.val_epses:
        attack_cls = getattr(attacks, val_attack_cls)
        attack = attack_cls(eps=val_eps / 255, proba=1)
        val_attacks.append(attack)

mean_mos = 0
images_in_dataset = 0
for _, __, mos in val_dataloader:
    mean_mos += mos.sum().item()
    images_in_dataset += mos.numel()
mean_mos /= images_in_dataset
print(f"mean mos: {mean_mos}")


for model_name in trained_models_names:
    model = lpips.LPIPS()
    model.load_state_dict(torch.load(os.path.join(opt.save_dir, model_name + ".pt")), strict=False)
    model.to(device)
    model.eval()
    result_correlations = np.zeros(len(val_attacks), dtype=np.float32)

    for i, val_attack in enumerate(val_attacks):
        print(f"evaluating model {model_name} on attack {val_attack.get_name()}")
        results = []
        targets = []
        for ref, dist, mos in tqdm(val_dataloader):
            ref = ref.to(device)
            dist = dist.to(device)
            mos = mos.to(device)
            better_than_mean = mos > mean_mos
            attack_direction = better_than_mean * -2 + 1
            attacked = val_attack.attack_impl(model, ref, dist, attack_direction)
            results_attacked = model.forward(ref, attacked).cpu().detach().numpy().flatten().tolist()
            results.extend(results_attacked)
            targets.extend((-mos).cpu().detach().numpy().flatten().tolist())
        res = stats.spearmanr(targets, results).statistic
        result_correlations[i] = res
        print(res)
    np.save(os.path.join(opt.experiment_dir, f"{model_name}_2kadid.npy"),  result_correlations)


