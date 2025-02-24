import os, argparse, torch, logging, warnings, sys

import numpy as np
from torch.utils.data import DataLoader
from data import CorpusSC
from utils.utils import evaluateNLI
from utils.logger import Logger
from model import BertMetaLearning
from datapath import get_loc

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="")
parser.add_argument("--hidden_dims", type=int, default=768, help="")

parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="name of the pretrained model",
)
parser.add_argument(
    "--local_model", action="store_true", help="use local pretrained model"
)
parser.add_argument("--sc_labels", type=int, default=3, help="")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--seed", type=int, default=0, help="seed for numpy and pytorch")
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="Print after every log_interval batches",
)
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--log_file", type=str, default="zeroshot_output.txt", help="")
parser.add_argument("--grad_clip", type=float, default=1.0)

parser.add_argument("--task", type=str, default="sc_fa")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))
print(args)

if torch.cuda.is_available():
    if not args.cuda:
        args.cuda = True
    torch.cuda.manual_seed_all(args.seed)
DEVICE = torch.device("cuda" if args.cuda else "cpu")


def load_data(task_lang):
    test_corpus = CorpusSC(
        *get_loc("test", task_lang, args.data_dir),
        model_name=args.model_name,
        local_files_only=args.local_model
    )
    batch_size = args.sc_batch_size

    return test_corpus, batch_size


test_corpus, batch_size = load_data(args.task)
test_dataloader = DataLoader(
    test_corpus, batch_size=batch_size, pin_memory=True, drop_last=True
)

if args.load != "":
    model = torch.load(args.load)
else:
    model = BertMetaLearning(args).to(DEVICE)


def test():
    model.eval()
    test_loss, test_acc, matrix = evaluateNLI(
        model, test_dataloader, DEVICE, return_matrix=True
    )
    print("test_loss {:10.8f} test_acc {:6.4f}".format(test_loss, test_acc))
    print("confusion matrix:\n", matrix)
    return test_loss


if __name__ == "__main__":
    test()
