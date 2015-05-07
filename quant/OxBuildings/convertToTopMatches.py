#!/usr/bin/python -tt

# computes top matches for every image, using a given a scoring

import os, sys, math, subprocess, random
import numpy as np
sys.path.append('../')
from computeScores_DCG import computeDCG
from nms import non_max_suppression_fast
sys.path.append('../learn/multi_patch_weights/')
from selectPatches import selectPatches
import h5py

takeTopN = 1 # -n = random n patches
              # -1 = 1 random patch
              # 1 = top match
              # 5 = top 5 matches
param1 = -0
upto = 1 # 0=> select nth. 1=> select 1..nth (only valid for top matches, not random)
nmsTh = 0.9 # set = -1 for no NMS
          # else, set a threshold between [0, 1]
N_OUTPUT = 9999999;

if 0:
  # for full img matching case
  method = 'full-img'
  get_class_style = 'oxford'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/Jegou13_hesaff_heatmap'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_top/Jegou13_hesaff_heatmap.txt'
  #simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS
elif 1:
  # for full img matching case
  method = 'full-img'
  get_class_style = 'oxford'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_refined/Jegou13_hesaff'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0008_OxBuildings/matches_top/Jegou13_hesaff.txt'
  #simsmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/learn/pairwise_matches_bin/'
  nmsTh = -1 # set = -1 for no NMS

MAXBOXPERIMG = 10000

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]
  
  fout = open(outfpath, 'w')
  allscores = np.zeros((1, 8))
  allscores_cls = {} # same as allscores, except separately for each cls
  numdists_cls = {}
  for i in testlist:
    print i
    
    selected = [0]
    matches = readMatches(matchesdir, i, selected)

    
    qboxes = [(i) * MAXBOXPERIMG + el + 1 for el in selected]
    fout.write('%s; ' % ','.join([str(el) for el in qboxes])) # query box
    for match in matches[:min(N_OUTPUT, len(matches))]:
      fout.write('%d:%f ' % (match[1] / MAXBOXPERIMG, match[0]))
    fout.write('\n')
  fout.close()

# outputs [(score, imid, imfeatids)...] // imid is not the imid*10K+featid
def readMatches(matchesdir, i, boxids):
  fpath = os.path.join(matchesdir, str(i) + '.txt')
  lines = readLines(fpath, boxids)
  allmatches = []
  for line in lines:
    matches = []
    for el in line.strip().split()[:N_OUTPUT]:
      el2 = el.split(':')
      matches.append((float(el2[1]), int(el2[0])))
    allmatches.append(matches)
  assert(len(allmatches) == 1)
  return allmatches[0]

def readLines(fpath, lnos): # lnos must be 0 indexed
  order = np.argsort(np.array(lnos))
  lines = []
  with open(fpath) as f:
    for i, line in enumerate(f):
      if i in lnos:
        lines.append(line)
  return list(np.array(lines)[order])

def getImgId(idx):
  return idx / MAXBOXPERIMG

if __name__ == '__main__':
  main()

