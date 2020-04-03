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
    model = DKT(cfg)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.solver.lr,
        momentum=cfg.solver.momentum,
        weight_decay=cfg.solver.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=1/1.5,
    )
    train_set = ASSIST(cfg, train=True)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.solver.batch_size,
        num_workers=cfg.solver.num_workers,
        shuffle=True,
        drop_last=False,
        collate_fn=my_collate_fn,
    )
    test_set = ASSIST(cfg, train=False)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.solver.batch_size,
        num_workers=cfg.solver.num_workers,
        shuffle=False,
        collate_fn=my_collate_fn,
    )

    for epoch in range(cfg.solver.epochs):
        pbar = tqdm(train_loader)
        model.train()
        for packed_xs, packed_skills, packed_answers in pbar:
            xs, lengths = rnn.pad_packed_sequence(packed_xs, padding_value=-1)
            skills, _ = rnn.pad_packed_sequence(packed_skills, padding_value=-1)
            answers, _ = rnn.pad_packed_sequence(packed_answers, padding_value=-1)

            loss = model(xs, skills, answers)

            pbar.set_description("Epoch: {}, Loss: {:.5f}".format(epoch + 1, float(loss)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        scheduler.step()
        
        model.eval()
        all_answers = []
        all_results = []
        for packed_xs, packed_skills, packed_answers in tqdm(test_loader):
            xs, lengths = rnn.pad_packed_sequence(packed_xs)
            skills, _ = rnn.pad_packed_sequence(packed_skills)
            answers, _ = rnn.pad_packed_sequence(packed_answers)

            with torch.no_grad():
                results = model(xs, skills, answers)
            
            answers = packed_answers.data
            results = rnn.pack_padded_sequence(results, lengths).data

            all_answers.append(answers)
            all_results.append(results)

        answers = torch.cat(all_answers).detach().cpu().numpy()
        results = torch.cat(all_results).detach().cpu().numpy()
        score = roc_auc_score(answers, results)
        print("AUC: {:.3f}".format(score))

        if (epoch + 1) % cfg.save_interval == 0:
            torch.save(model.state_dict(), "epoch-{}.pth".format(epoch + 1))




            



if __name__ == "__main__":
    main()
