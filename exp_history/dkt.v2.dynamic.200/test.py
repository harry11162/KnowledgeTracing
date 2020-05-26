import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn

from config import cfg
from network import DKT

from lib.dataset import ASSIST, my_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", "-e", type=int)
    args = parser.parse_args()

    model = DKT(cfg)
    model.load_state_dict(torch.load("epoch-{}.pth".format(args.epoch)))
    model.eval()

    test_set = ASSIST(cfg, train=False)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.solver.batch_size,
        num_workers=cfg.solver.num_workers,
        shuffle=False,
        collate_fn=my_collate_fn,
    )
    all_answers = []
    all_results = []
    for packed_xs, packed_skills, packed_answers in tqdm(test_loader):
        xs, lengths = rnn.pad_packed_sequence(packed_xs)
        skills, _ = rnn.pad_packed_sequence(packed_skills)
        answers, _ = rnn.pad_packed_sequence(packed_answers)

        with torch.no_grad():
            results = model(xs, skills, answers)

        answers = packed_answers.data
        skills = packed_skills.data
        results = rnn.pack_padded_sequence(results, lengths).data
        results = results.gather(-1, skills[:, None]).squeeze(-1)

        all_answers.append(answers)
        all_results.append(results)

    answers = torch.cat(all_answers).detach().cpu().numpy()
    results = torch.cat(all_results).detach().cpu().numpy()
    score = roc_auc_score(answers, results)
    print("AUC: {:.3f}".format(score))


if __name__ == "__main__":
    main()
