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
import pdb

takeTopN = 1 # -n = random n patches
              # -1 = 1 random patch
              # 1 = top match
              # 5 = top 5 matches
param1 = -0
upto = 1 # 0=> select nth. 1=> select 1..nth (only valid for top matches, not random)
nmsTh = 0.9 # set = -1 for no NMS
          # else, set a threshold between [0, 1]
use_similarity_selection = False # = True for using the similarity scores for multi patch sel
N_OUTPUT = 99999999;
NMATCHES_PER_PATCH = 999999;

# for the grid search
method = 'full-img'
matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/train_GridSearch/david_FaceHeatmap/'
#matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/train_GridSearch/crossval/'
retrievallistpath =  '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesTrain_noNeg.txt'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/Jegou13_hesaff_heatmap_gridsearch.txt'
simsmatdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches/'
nmsTh = -1 # set = -1 for no NMS


MAXBOXPERIMG = 10000

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]
  with open(retrievallistpath) as f:
    retlist = [int(t) for t in f.read().splitlines()]

  print 'mp1,mp3,mp5,mp10,mp20,atleast1/3,atleast1/10,DCG/10,mAP'
  threshes = np.arange(-0.1, 1, 0.1)
  allscores = np.zeros((len(threshes), 9))
  thi = 0
  for th in threshes:
    fout = open(outfpath, 'w')
    for i in testlist:
      # get the top matches from each and intersection
      selected = [0]
      thdirname = str(th)
      if th == 0:
        thdirname = str(0)
      matches = readMatches(matchesdir + '/' + thdirname + '/', i, selected, retlist)
      
      matches = randomSortZeroScores(matches)

      scores = computeScores(matches, i, imgslist)
      allscores[thi][:-1] += np.array(scores)
      
      qboxes = [(i) * MAXBOXPERIMG + el + 1 for el in selected]
      fout.write('%s; ' % ','.join([str(el) for el in qboxes])) # query box
      for match in matches[:min(N_OUTPUT, len(matches))]:
        fout.write('%d:%f:%s ' % (match[1], match[0], 
              ','.join([str(el) for el in match[2]])))
      fout.write('\n')
    fout.close()
    # compute mAP
    cmd = 'cd /srv2/rgirdhar/Work/Code/0003_DiscoverPatches/DiscoverPatches/quant/OxBuildings/;  ./run_scripts/runap_pal_train.sh ' + os.path.basename(outfpath)
    mAPScore = subprocess.check_output(cmd, shell=True)
    mAPScore = float(mAPScore.split(':')[1].strip())
    allscores[thi][-1] += mAPScore * len(testlist) # because later it gets divided
 
    if th < 0:
      print 'mean::'
    print ','.join([str(s) for s in (allscores[thi] / len(testlist)).tolist()])
    thi += 1


# outputs [(score, imid, imfeatids)...] // imid is not the imid*10K+featid
def readMatches(matchesdir, i, boxids, retlist):
  fpath = os.path.join(matchesdir, str(i) + '.txt')
  lines = readLines(fpath, boxids)
  allmatches = []
  for line in lines:
    matches = []
    line_matches =  line.strip().split()
    for el in line_matches[:min(NMATCHES_PER_PATCH, len(line_matches))]:
      el2 = el.split(':')
      matches.append((float(el2[1]), int(el2[0])))
    allmatches.append(matches)
  matches = mergeRanklists(allmatches, retlist)
  return matches

# outputs [(score, imid, imfeatids)...] // imid is not the imid*10K+featid
def readMatchesWithFull(matchesdir, fullmatchesdir, i, boxids, FULL_MATCH_WT, retlist):
  # patch matches
  fpath = os.path.join(matchesdir, str(i) + '.txt')
  lines = readLines(fpath, boxids)
  allmatches = []
  for line in lines:
    matches = []
    line_matches =  line.strip().split()
    for el in line_matches[:min(NMATCHES_PER_PATCH, len(line_matches))]:
      el2 = el.split(':')
      matches.append((float(el2[1]), int(el2[0])))
    allmatches.append(matches)
  # full matches
  fpath = os.path.join(fullmatchesdir, str(i) + '.txt')
  lines = readLines(fpath, [0])
  for line in lines:
    matches = []
    line_matches =  line.strip().split()
    for el in line_matches[:min(NMATCHES_PER_PATCH, len(line_matches))]:
      el2 = el.split(':')
      matches.append((float(el2[1]) * FULL_MATCH_WT, int(el2[0])))
    allmatches.append(matches)

  matches = mergeRanklists(allmatches, retlist)
  return matches

# returns [(score, imgid, imfeatids)..]
# retlist is the list of ids all images in the corpus from which to retrieve
def mergeRanklists(allmatches, retlist):
  imid2score = {}
  imid2feats = {} # store what bounding boxes in this image matched

  # initialize the lists
  for retel in retlist:
    imid2score[retel] = 0
    imid2feats[retel] = []

  for matches in allmatches:
    for match in matches:
      imid = getImgId(match[1])
      #if imid not in imid2score.keys():
      #  imid2score[imid] = match[0]
      #  imid2feats[imid] = [match[1]]
      #else:
      imid2score[imid] += match[0]
      imid2feats[imid].append(match[1])
  res = imid2score.items()
  res = sorted(res, key=lambda tup: tup[1], reverse=True) # remember, reverse sort!
  res = [(m[1], m[0], imid2feats[m[0]]) for m in res]
  return res

# matches must be [(score, imid)...]
def computeScores(matches, imgid, imgslist):
  # remove the exact match
  matches2 = matches[:]
  sameornot = [m[1] == imgid for m in matches2]
  if sum(sameornot) > 0:
    del matches2[np.where(sameornot)[0][0]]

  clses = [getClass(m[1] - 1, imgslist) for m in matches2]
  cls = getClass(imgid - 1, imgslist)
  hits = [c == cls for c in clses]
  scores = []
  for i in [1,3,5,10,20]: # for mP
    scores.append(float(sum(hits[:i])) / i)
  scores.append(sum(hits[:3]) > 0) # for atleast 1/3
  scores.append(sum(hits[:10]) > 0)
  scores.append(computeDCG(hits[:10], 10))
  return scores

# matches must be [(score, imid)...]
def countMatchesOfClass(matches, imgslist, n, cls):
  clses = [getClass(m[1] - 1, imgslist) for m in matches][:n]
  hits = [c == cls for c in clses]
  return sum(hits)

def readLines(fpath, lnos): # lnos must be 0 indexed
  order = np.argsort(np.array(lnos))
  lines = []
  with open(fpath) as f:
    for i, line in enumerate(f):
      if i in lnos:
        lines.append(line)
  return list(np.array(lines)[order])

def getClass(imid, imgslist): 
  # imgid here must be 0-indexed!!!
  try:
    if get_class_style == 'oxford':
      return '_'.join(imgslist[imid].split('_')[:-1])
    else:
      return os.path.dirname(imgslist[imid])
  except NameError: # get_class_style not defined
    return os.path.dirname(imgslist[imid])

def getImgId(idx):
  return idx / MAXBOXPERIMG

def performNMS(order, selboxfpath, th):
  f = open(selboxfpath)
  boxes = [[float(el) for el in line.split(',')] 
      for line in f.read().splitlines()]
  f.close()
  # this order is descending, so store reverse of this order for nms
  # and then reverse back the output
  boxes = [(box[1],box[0],box[3],box[2]) for box in boxes]
  boxes_rev = []
  order_rev = []
  for i in order[::-1]:
    boxes_rev.append(boxes[i])
    order_rev.append(i)
  _, pick = non_max_suppression_fast(np.array(boxes_rev), th)
  order_rev = np.array(order_rev)
  return order_rev[pick]

def readHDF5(fpath, dbname):
  print fpath
  f = h5py.File(fpath, 'r')
  data = f[dbname][:]
  f.close()
  return data

def randomSortZeroScores(matches):
  random.seed(1)
  actual_matches = [el for el in matches if el[0] > 0]
  other_matches = [el for el in matches if el[0] == 0]
  random.shuffle(other_matches)
  actual_matches += other_matches
  return actual_matches

if __name__ == '__main__':
  main()

