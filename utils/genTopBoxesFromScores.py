import numpy as np
import os

scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/query_scores/fc7_PeopleOnly/'
testidxfpath = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/query_scores/fc7_PeopleOnly_topIdxs'
N = 100

testidx = [int(el) for el in 
  open(testidxfpath).read().splitlines()]
for i in testidx:
  sc = np.loadtxt(os.path.join(scoresdir, str(i) + '.txt'))
  idxs = np.argsort(-sc) + 1 # to get 1 indexed
  outfpath = os.path.join(outdir, str(i) + '.txt')
  np.savetxt(outfpath, idxs[:N], delimiter='\n', fmt='%d')

