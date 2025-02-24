import torch
import argparse
import numpy as np
import logging
from torch.utils.data import DataLoader
from data import CorpusSC
from utils.utils import evaluateNLI
import os

import warnings

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
from datapath import loc

parser = argparse.ArgumentParser()

parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--tc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--po_batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--pa_batch_size", type=int, default=8, help="batch size")

parser.add_argument("--seed", type=int, default=42, help="seed for numpy and pytorch")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--eval_tasks", type=str, default="qa,sc,pa,po,tc")

parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

if torch.cuda.is_available():
    if not args.cuda:
        # print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        args.cuda = True

    torch.cuda.manual_seed_all(args.seed)

DEVICE = torch.device("cuda" if args.cuda else "cpu")

task_types = args.eval_tasks.split(",")
list_of_tasks = []

for tt in loc["train"].keys():
    if tt[:2] in task_types:
        list_of_tasks.append(tt)

num_tasks = len(list_of_tasks)
print(list_of_tasks)

test_corpus = {}

for k in list_of_tasks:
    test_corpus[k] = CorpusSC(loc["test"][k][0], loc["test"][k][1])

test_dataloaders = {}

for k in list_of_tasks:
    batch_size = (
        args.qa_batch_size
        if "qa" in k
        else args.sc_batch_size
        if "sc" in k
        else args.tc_batch_size
        if "tc" in k
        else args.po_batch_size
        if "po" in k
        else args.pa_batch_size
    )
    test_dataloaders[k] = DataLoader(
        test_corpus[k], batch_size=batch_size, pin_memory=True, drop_last=True
    )

is_single = False
is_baseline = False

if os.path.exists(os.path.join(args.load, "model.pt")):
    is_baseline = True
else:
    is_baseline = False
best_results = {k: float("-inf") for k in list_of_tasks}

if args.load.count(".pt") > 0:
    model = torch.load(args.load)

    for k in list_of_tasks:
        test_loss, test_acc = evaluateNLI(model, test_dataloaders[k], DEVICE)
        best_results[k] = max(best_results[k], test_acc)

elif os.path.exists(os.path.join(args.load, "model.pt")):
    model = torch.load(os.path.join(args.load, "model.pt"))

    for k in list_of_tasks:
        test_loss, test_acc = evaluateNLI(model, test_dataloaders[k], DEVICE)
        best_results[k] = max(best_results[k], test_acc)

else:
    model_found = []

    for saved_model in os.listdir(args.load):
        if saved_model.find(".pt") > 0:
            model_task = saved_model.split(".")[0].split("_")[1]
            model_found.append(model_task)

    for saved_model in os.listdir(args.load):
        if saved_model.find(".pt") > 0:
            model_task = saved_model.split(".")[0].split("_")[1]
            model = torch.load(os.path.join(args.load, saved_model))
            model.eval()
            for k in list_of_tasks:
                test_loss, test_acc = evaluateNLI(model, test_dataloaders[k], DEVICE)
                best_results[k] = max(best_results[k], test_acc)

            print(saved_model, best_results)

print(args.load)
for k in list_of_tasks:
    print(k, "{:6.4f}".format(best_results[k]))
