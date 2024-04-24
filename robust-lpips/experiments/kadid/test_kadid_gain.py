import numpy as np
import torch
import lpips
from tqdm import tqdm
import os
from experiments import parser
from experiments.kadid import kadid_parser
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


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
                    attack_direction = torch.ones(len(ref)).to(opt.device)
                    attacked = val_attack.attack_impl(model, ref, dist, attack_direction)
                    results_attacked.extend(model.forward(ref, attacked).cpu().detach().numpy().flatten().tolist())

                all_attack_results[model_name][attack_name] = results_attacked

        for val_attack in opt.val_attacks:
            attack_name = val_attack.get_name()
            if attack_name == opt.dummy_attack_name:
                continue
            plt.clf()
            for model_name in opt.trained_models_names:
                clean_results = np.array(all_attack_results[model_name][opt.dummy_attack_name]).reshape(-1, 1)
                attack_results = np.array(all_attack_results[model_name][attack_name]).reshape(-1, 1)
                scaler = MinMaxScaler()
                clean_results = scaler.fit_transform(clean_results)
                attack_results = scaler.transform(attack_results)
                gain = clean_results - attack_results
                avg_gain = np.mean(gain)
                plt.scatter(clean_results, gain, label=f"{model_name}, avg gain = {avg_gain:.3f}", s=1)
            plt.grid()
            plt.title(f"gains for attack {attack_name} on group {group} (lower is better)")
            plt.xlabel("clean results")
            plt.ylabel("absolute gain")
            plt.legend()
            plt.savefig(os.path.join(opt.experiment_dir, f"kadid_avg_gain_{attack_name}_{group}.png"))


if __name__ == "__main__":
    default_parser = parser.make_default_parser()
    default_parser = kadid_parser.add_groups(default_parser)

    opt = default_parser.parse_args()
    opt.test_no_attack = True
    opt = parser.setup_options(opt)

    test_kadid(opt)
