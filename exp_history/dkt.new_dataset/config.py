import sys
from easydict import EasyDict as edict

sys.path.append("../..")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cfg = edict()

cfg.data = edict()
cfg.data.train_file = "/mnt/data/haomiao/DeepKnowledgeTracing/data/assistments/builder_train.csv"
cfg.data.test_file = "/mnt/data/haomiao/DeepKnowledgeTracing/data/assistments/builder_test.csv"
cfg.data.input_size = 124 * 2

cfg.solver = edict()
cfg.solver.batch_size = 32
cfg.solver.num_workers = 1
cfg.solver.epochs = 100
cfg.solver.lr = 0.05
cfg.solver.momentum = 0.9
cfg.solver.weight_decay = 0

cfg.model = edict()
cfg.model.hidden_size = 200

cfg.save_interval = 20
