import numpy as np
import tqdm
from common.attacks import NoAttack


def score_2afc_dataset(data_loader, model, device, attack=None):
    if attack is None:
        attack = NoAttack()

    d0s = []
    d1s = []
    gts = []

    model.eval()
    model.to(device=device)
    for data in tqdm.tqdm(data_loader.load_data(), desc=attack.get_name()):
        ref = data['ref'].to(device=device)
        p0 = data['p0'].to(device=device)
        p1 = data['p1'].to(device=device)
        input_judge = data['judge']
        attacked_p0, attacked_p1 = attack.attack_pairs(model, ref, p0, p1, input_judge)
        d0s += model.forward(ref, attacked_p0).cpu().detach().numpy().flatten().tolist()
        d1s += model.forward(ref, attacked_p1).cpu().detach().numpy().flatten().tolist()
        gts += input_judge.cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    print(d0s.shape, d1s.shape, gts.shape)
    outputs = np.stack((d1s, d0s), axis=1)
    correct = outputs.argmax(1) == np.round(gts)
    scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
    print(np.mean(correct))
    return np.mean(scores), np.mean(correct)
