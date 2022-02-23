from policyGradientNetwork import *
from game import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-A', '--a', type=int)
args = parser.parse_args()
print(args.a)


