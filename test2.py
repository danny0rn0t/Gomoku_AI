import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a", action='store_true')
parser.add_argument("--b", action='store_true')

args = parser.parse_args()
print(args.a)
print(args.b)
