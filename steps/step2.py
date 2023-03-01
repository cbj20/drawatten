import pickle
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", type=str, default="/home/nfs_data/zhanggh/drawatten/tmp/mrpc/input_ids", help="Path to input_ids")
  args = parser.parse_args()

  with open(args.path, "rb") as fp:
    b = pickle.load(fp)
    print(len(b))

if __name__ == '__main__':
  main()