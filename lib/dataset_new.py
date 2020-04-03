import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn

import csv

class ASSIST(Dataset):
    def __init__(self, cfg, train=True):
        data_file = cfg.data.train_file if train else cfg.data.test_file
        self.data = self.read_file(data_file)

    def __getitem__(self, index):
        data = self.data[index]
        skills = torch.tensor(data["skills"], dtype=torch.long)
        answers = torch.tensor(data["answers"], dtype=torch.long)

        L = skills.size()[0]
        C = 124
        combined = skills + answers * C

        x = torch.zeros((L, 2 * C), dtype=torch.float32)
        x.scatter_(dim=1, index=combined[:, None], value=1.0)

        return x, skills, answers


    def __len__(self):
        return len(self.data)

    def read_file(self, data_file):
        data = []
        count = [0, 0]
        with open(data_file, "r", newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.length = 0
            while True:
                try:
                    num_q = next(reader)
                    skills = next(reader)
                    answers = next(reader)

                    if skills[-1] == '': skills.pop()
                    if answers[-1] == '': answers.pop()

                    num_q = int(num_q[0])
                    skills = [int(i) for i in skills]
                    answers = [int(i) for i in answers]

                    assert num_q == len(skills)
                    assert num_q == len(answers)

                    num_correct = sum(answers)
                    num_answers = len(answers)
                    count[0] += num_correct
                    count[1] += num_answers
                    data_per_student = {"skills": skills,
                                        "answers": answers}
                    data.append(data_per_student)
                except:
                    break

        print("{} Answers in Total! {} of them Correct!".format(count[1], count[0]))

        return data

def my_collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    batch = list(zip(*batch))

    xs, skills, answers = batch
    xs = rnn.pack_sequence(xs)
    skills = rnn.pack_sequence(skills)
    answers = rnn.pack_sequence(answers)
    return xs, skills, answers


if __name__ == "__main__":
    dataset = ASSIST(1)
