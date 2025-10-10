import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="./test_output")
args = parser.parse_args()

print("code excuteted:")
print("args:")
print("epochs:", args.epochs)
print("batch_size:", args.batch_size)
print("save_dir", args.save_dir)

keys = vars(args).keys()
print(list(keys))