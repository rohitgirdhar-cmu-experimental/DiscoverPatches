import os, sys, math, subprocess, random
import numpy as np
import h5py

simmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
simmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/train_crossval_scores/'
scoresdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/train_crossval_scores_bin/'
trainidxs = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'

with open(trainidxs) as f:
  testlist = [int(t) for t in f.read().splitlines()]

def main():
  print evalParamValue(0.9)

def evalParamValue(param):
  tot = 0
  for i in testlist:
    patchscores = np.loadtxt(os.path.join(scoresdir, str(i) + '.txt'))
    sims = np.loadtxt(os.path.join(simmatdir, str(i) + '.txt'))
    f = h5py.File(os.path.join(simmatdir_bin, str(i) + '.h5'), 'w')
    f.create_dataset("sims", data=sims)
    f.close()
    f = h5py.File(os.path.join(scoresdir_bin, str(i) + '.h5'), 'w')
    f.create_dataset("scores", data=patchscores)
    f.close()

if __name__ == '__main__':
  main()

