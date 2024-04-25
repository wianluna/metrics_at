import numpy as np
import torch
import lpips
from tqdm import tqdm
import os
from scipy import stats
from experiments import parser
from experiments.kadid import kadid_parser
import matplotlib.pyplot as plt
from collections import defaultdict


def test_kadid(opt):
    for val_dataloader, group in kadid_parser.iterate_dataloaders(opt):
        print(f"testing group {group}")
        all_attack_results = defaultdict(dict)
        mean_mos = 0
        images_in_dataset = 0
        targets = []
        for _, __, mos in val_dataloader:
            mean_mos += mos.sum().item()
            images_in_dataset += mos.numel()
            targets.extend((-mos).numpy().flatten().tolist())

        mean_mos /= images_in_dataset
        for model_name in opt.trained_models_names:
            model = lpips.LPIPS()
            model.load_state_dict(torch.load(os.path.join(opt.models_dir, model_name + ".pt")), strict=False)
            model.to(opt.device)
            model.eval()

            for val_attack in opt.val_attacks:
                attack_name = val_attack.get_name()
                plt.clf()
                print(f"evaluating model {model_name} on attack {attack_name}")
                results_attacked = []

                for ref, dist, mos in tqdm(val_dataloader):
                    ref = ref.to(opt.device)
                    dist = dist.to(opt.device)
                    mos = mos.to(opt.device)
                    better_than_mean = mos > mean_mos
                    attack_direction = better_than_mean * -2 + 1
                    attacked = val_attack.attack_impl(model, ref, dist, attack_direction)
                    results_attacked.extend(model.forward(ref, attacked).cpu().detach().numpy().flatten().tolist())

                srocc = stats.spearmanr(targets, results_attacked).statistic
                all_attack_results[model_name][attack_name] = (results_attacked, srocc)

    for val_attack in opt.val_attacks:
        attack_name = val_attack.get_name()
        plt.clf()
        for model_name in opt.trained_models_names:
            results_attacked, srocc = all_attack_results[model_name][attack_name]
            plt.scatter(targets, results_attacked, label=f"{model_name}, SROCC = {srocc:.3f}", s=1)

        plt.grid()
        plt.title(f"correlations for attack {attack_name} on group {group} (higher is better)")
        plt.xlabel("targets")
        plt.ylabel("attacked results")
        plt.legend()
        plt.savefig(os.path.join(opt.experiment_dir, f"kadid_srocc_{attack_name}_{group}.png"))


if __name__ == "__main__":
    default_parser = parser.make_default_parser()
    default_parser = kadid_parser.add_groups(default_parser)

    opt = default_parser.parse_args()
    opt = parser.setup_options(opt)

    test_kadid(opt)
