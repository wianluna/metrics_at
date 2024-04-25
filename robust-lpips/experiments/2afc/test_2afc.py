import numpy as np
import torch
import lpips
import os
from common.data.bapps.data_loader import CreateDataLoader
from common.score_2afc_dataset import score_2afc_dataset
from experiments import parser


def test_2afc(opt):
    val_dataloader = CreateDataLoader(
        opt.val_datasets,
        data_root=opt.dataroot,
        batch_size=opt.batch_size, 
        serial_batches=True,
        nThreads=opt.num_workers
    )

    for model_name in opt.trained_models_names:
        model = lpips.LPIPS()
        model.load_state_dict(torch.load(os.path.join(opt.models_dir, model_name + ".pt")), strict=False)
        result_2afc = np.zeros(len(opt.val_attacks), dtype=np.float32)
        result_correct = np.zeros_like(result_2afc)

        for i, val_attack in enumerate(opt.val_attacks):
            print(f"evaluating model {model_name} on attack {val_attack.get_name()}")
            score_2afc, correct_proportion = score_2afc_dataset(val_dataloader, model, opt.device, val_attack)
            result_2afc[i] = score_2afc
            result_correct[i] = correct_proportion
            print(f"results: 2afc score = {score_2afc}, correct proportion = {correct_proportion}")

        np.save(os.path.join(opt.experiment_dir, f"{model_name}_2afc.npy"),  result_2afc)
        np.save(os.path.join(opt.experiment_dir, f"{model_name}_correct.npy"),  result_correct)


if __name__ == "__main__":
    default_parser = parser.make_default_parser()
    default_parser.add_argument('--val_datasets', nargs='+', default=['val/cnn','val/color', 'val/deblur', 'val/frameinterp', 'val/superres'], help='datasets to validate model on')
    opt = default_parser.parse_args()
    opt = parser.setup_options(opt)
    test_2afc(opt)
