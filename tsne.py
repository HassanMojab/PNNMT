import os, argparse, torch, logging, warnings, sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from data import CorpusSC
from utils.logger import Logger
from datapath import loc, get_loc

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--num_labels", type=int, default=3, help="number of labels")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--steps", type=int, default=32, help="total steps")
parser.add_argument("--seed", type=int, default=0, help="seed for numpy and pytorch")
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="Print after every log_interval batches",
)
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--tpu", action="store_true", help="use TPU")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="saved/model_last.pt", help="")
parser.add_argument("--log_file", type=str, default="tsne.txt", help="")
parser.add_argument("--tasks", type=str, default="sc")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))
print(args)

task_types = args.meta_tasks.split(",")
list_of_tasks = []

for tt in loc["train"].keys():
    if tt[:2] in task_types:
        list_of_tasks.append(tt)

for tt in task_types:
    if "_" in tt:
        list_of_tasks.append(tt)

list_of_tasks = list(set(list_of_tasks))
print(list_of_tasks)

if torch.cuda.is_available():
    if not args.cuda:
        args.cuda = True

    torch.cuda.manual_seed_all(args.seed)

# DEVICE = xm.xla_device() if args.tpu else torch.device("cuda" if args.cuda else "cpu")
DEVICE = torch.device("cuda" if args.cuda else "cpu")


dataloaders = []

for task in list_of_tasks:
    corpus = CorpusSC(
        *get_loc("train", task, "data"),
        model_name="xlm-roberta-base",
        local_files_only=False,
    )
    dataloader = DataLoader(corpus, batch_size=args.batch_size, shuffle=True)
    dataloaders.append(dataloader)

model = torch.load(args.load)


def main():
    features_list = []
    labels_list = []

    total_steps = args.steps

    model.eval()
    with torch.no_grad():
        for task, dataloader in zip(list_of_tasks, dataloaders):
            print(f"task {task} --------------------------------")
            for i, batch in enumerate(dataloader):
                if i >= total_steps:
                    break

                labels_list.append(batch["label"])
                _, features = model.forward(batch, classify=False)
                features_list.append(features.cpu().detach().numpy())

                if (i + 1) % 10 == 0:
                    print(f"batch#{i + 1} | {(i + 1) / total_steps * 100:.2f}%")

    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)

    tsne = TSNE()
    X_embedded = tsne.fit_transform(features)
    sns.set(rc={"figure.figsize": (11.7, 8.27)})
    palette = sns.color_palette("bright", args.num_labels)

    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=labels,
        legend="full",
        palette=palette,
    )
    plt.show()
    plt.savefig(os.path.join(args.save, "tsne.png"))


if __name__ == "__main__":
    main()
