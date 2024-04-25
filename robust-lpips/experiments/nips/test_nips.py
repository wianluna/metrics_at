import torch
import lpips
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from experiments import parser
from common.data.nips.nips import CustomImageDataset


def test_nips(opt):
    val_dataloader = torch.utils.data.DataLoader(CustomImageDataset(opt.dataroot), batch_size=opt.batch_size, num_workers=opt.num_workers)
    all_attack_results = defaultdict(dict)
    for model_name in opt.trained_models_names:
        model = lpips.LPIPS()
        model.load_state_dict(torch.load(os.path.join(opt.models_dir, model_name + ".pt")), strict=False)
        model.to(opt.device)
        model.eval()
        for val_attack in opt.val_attacks:
            print(f"evaluating model {model_name} on attack {val_attack.get_name()}")
            attack_results = []
            for ref, dist, in tqdm(val_dataloader):
                ref = ref.to(opt.device)
                dist = dist.to(opt.device)

                attack_direction = torch.ones(len(ref)).to(opt.device)
                attacked = val_attack.attack_impl(model, ref, dist, attack_direction)
                batch_results = model.forward(ref, attacked).cpu().detach().numpy().flatten().tolist()
                attack_results.extend(batch_results)
            all_attack_results[model_name][val_attack.get_name()] = attack_results

    for val_attack in opt.val_attacks:
        attack_name = val_attack.get_name()
        if attack_name == opt.dummy_attack_name:
            continue
        plt.clf()
        for model_name in opt.trained_models_names:
            clean_results = all_attack_results[model_name][opt.dummy_attack_name]
            attack_results = all_attack_results[model_name][attack_name]
            scaler = MinMaxScaler()
            clean_results = scaler.fit_transform(clean_results)
            attack_results = scaler.transform(attack_results)
            gain = clean_results - attack_results
            avg_gain = np.mean(gain)
            plt.scatter(clean_results, gain, label=f"{model_name}, avg gain = {avg_gain:.3f}", s=1)
        plt.grid()
        plt.title(f"gains for attack {attack_name} (lower is better)")
        plt.xlabel("clean results")
        plt.ylabel("absolute gain")
        plt.legend()
        plt.savefig(os.path.join(opt.experiment_dir, f"nips_{attack_name}.png"))


if __name__ == "__main__":
    default_parser = parser.make_default_parser()
    opt = default_parser.parse_args()
    opt.test_no_attack = True
    opt = parser.setup_options(opt)
    test_nips(opt)
