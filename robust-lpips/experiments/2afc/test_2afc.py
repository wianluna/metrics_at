import numpy as np
import torch
import lpips
import argparse
import os
from common.data.bapps.data_loader import CreateDataLoader
from common.attacks import NoAttack
from common import attacks
from common.score_2afc_dataset import score_2afc_dataset


def test_2afc(opt):

    val_dataloader = CreateDataLoader(opt.val_datasets, data_root=opt.dataroot, batch_size=opt.batch_size, serial_batches=True, nThreads=opt.num_workers)

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

    for model_name in trained_models_names:
        model = lpips.LPIPS()
        model.load_state_dict(torch.load(os.path.join(opt.save_dir, model_name + ".pt")), strict=False)
        result_2afc = np.zeros(len(val_attacks), dtype=np.float32)
        result_correct = np.zeros_like(result_2afc)

        for i, val_attack in enumerate(val_attacks):
            print(f"evaluating model {model_name} on attack {val_attack.get_name()}")
            score_2afc, score_correct = score_2afc_dataset(val_dataloader, model, val_attack)
            result_2afc[i] = score_2afc
            result_correct[i] = score_correct
            print(score_2afc, score_correct)
        np.save(os.path.join(opt.experiment_dir, f"{model_name}_2afc.npy"),  result_2afc)
        np.save(os.path.join(opt.experiment_dir, f"{model_name}_correct.npy"),  result_correct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_datasets', type=str, nargs='+', default=['val/cnn','val/color', 'val/deblur', 'val/frameinterp', 'val/superres'], help='datasets to validate model on')
    parser.add_argument('--dataroot', type=str, default=".", help='dataset root')
    parser.add_argument('--save_dir', type=str, default="./models", help='directory where the trained models are')
    parser.add_argument('--experiment_dir', type=str, default="./bapps_stats", help='path to save experiment results')
    parser.add_argument('--train_attacks', type=str, default=[], help='attack types used for training')
    parser.add_argument('--train_epses', type=str, default=[2, 4, 8, 10], help='attack epsilons used for training')
    parser.add_argument('--val_attacks', type=str, default=["FGSM", "FreeFGSM", "PGD"], help='attack types for validation')
    parser.add_argument('--val_epses', type=str, default=[2, 4, 6, 8, 10], help='attack epsilons for validation')
    parser.add_argument('--train_dummy', action='store_true', help="evaluate model which wasn't adversarily trained")
    parser.add_argument('--test_dummy', action='store_true', help="validate model response without attacks")
    parser.add_argument('--test_r_lpips', action='store_true', help="validate R-LPIPS")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers')
    opt = parser.parse_args()

    test_2afc(opt)
