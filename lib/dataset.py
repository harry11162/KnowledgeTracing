import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn

import csv


all_skills = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40,
42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 79, 80,
81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
101, 102, 104, 105, 107, 112, 113, 115, 118, 119, 120, 121, 123]


class ASSIST(Dataset):
    def __init__(self, cfg, train=True):
        data_file = cfg.data.train_file if train else cfg.data.test_file
        self.data = self.read_file(data_file)

    def __getitem__(self, index):
        data = self.data[index]
        skills = torch.tensor(data["skills"], dtype=torch.long)
        answers = torch.tensor(data["answers"], dtype=torch.long)

        L = skills.size()[0]
        C = len(all_skills)
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
                    num_q, student_id = (int(i) for i in next(reader))
                    skills = [int(i) for i in next(reader)]
                    questions = [int(i) for i in next(reader)]
                    answers = [int(i) for i in next(reader)]
                    num_correct = sum(answers)
                    num_answers = len(answers)
                    count[0] += num_correct
                    count[1] += num_answers
                    data_per_student = {"skills": skills,
                                        "questions": questions,
                                        "answers": answers,
                                        "id": student_id, }
                    data.append(data_per_student)
                except:
                    break
            
        print("{} Answers in Total! {} of them Correct!".format(count[1], count[0]))
        
        # squeeze skills
        skill_map = {}
        for i, s in enumerate(all_skills):
            skill_map[s] = i
        for i in range(len(data)):
            data[i]["skills"] = [skill_map[j] for j in data[i]["skills"]]

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
