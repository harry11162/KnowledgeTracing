import sys
from easydict import EasyDict as edict

sys.path.append("../..")

cfg = edict()

cfg.data = edict()
cfg.data.train_file = "/mnt/data/haomiao/4_Ass_09_train.csv"
cfg.data.test_file = "/mnt/data/haomiao/4_Ass_09_test.csv"
cfg.data.input_size = 104 * 2

cfg.solver = edict()
cfg.solver.batch_size = 1000
cfg.solver.num_workers = 1
cfg.solver.epochs = 100
cfg.solver.lr = 0.05
cfg.solver.momentum = 0.9
cfg.solver.weight_decay = 0

cfg.model = edict()
cfg.model.hidden_size = 100

cfg.save_interval = 2
