import argparse
import torch
import os

from common import attacks


def make_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default=".", help="dataset root")

    parser.add_argument("--models_dir", type=str, default="./models", help="directory where the trained models are")
    parser.add_argument("--experiment_dir", type=str, default="./kadid_stats", help="path to save experiment results")

    parser.add_argument("--train_attacks", nargs='+', default=["FGSM"], help="attack types used for training")
    parser.add_argument("--train_epses", nargs='+', default=["4"], help="attack epsilons used for training")

    parser.add_argument("--val_attacks", nargs='+', default=["IFGSM"], help="attack types to run validation on")
    parser.add_argument("--val_epses", nargs='+', default=["10"], help="attack epsilons to run validation on")

    parser.add_argument("--test_baseline", action="store_true", help="evaluate model which wasn't adversarily trained")
    parser.add_argument("--test_r_lpips", action="store_true", help="validate R-LPIPS")

    parser.add_argument("--test_no_attack", action="store_true", help="validate models without an attack")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--device", type=str, default=None, help="torch device to run experiments on")
    return parser


def setup_options(opt):

    device_name = opt.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = torch.device(device_name)

    trained_models_names = []
    val_attacks = []

    if opt.test_baseline:
        trained_models_names.append("NoAttack")

    if opt.test_r_lpips:
        trained_models_names.append("latest_net_linf_x0")

    if opt.test_no_attack:
        dummy_attack = attacks.NoAttack()
        opt.dummy_attack_name = dummy_attack.get_name()
        val_attacks.append(dummy_attack)

    for train_attack_cls in opt.train_attacks:
        for train_eps in opt.train_epses:
            attack_cls = getattr(attacks, train_attack_cls)
            attack = attack_cls(eps=int(train_eps) / 255, proba=0.5)
            trained_models_names.append(attack.get_name())

    for val_attack_cls in opt.val_attacks:
        for val_eps in opt.val_epses:
            attack_cls = getattr(attacks, val_attack_cls)
            attack = attack_cls(eps=int(val_eps) / 255, proba=1)
            val_attacks.append(attack)

    opt.trained_models_names = trained_models_names
    opt.val_attacks = val_attacks
    if not os.path.exists(opt.experiment_dir):
        os.mkdir(opt.experiment_dir)
    return opt
