#!/usr/bin/python -tt

# computes top matches for every image, using a given a scoring

import os, sys, math, subprocess
import numpy as np

matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined/'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
boxesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/'
method = 'rbf_1K'
#method = 'gt'
scoresdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/query_scores_' + method
outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals/' + method + '.txt'

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]
  
  fout = open(outfpath, 'w')
  allscores = np.zeros((1, 7))
  for i in testlist:
    with open(os.path.join(scoresdir, str(i) + '.txt')) as f:
      scores = [float(el) for el in f.read().splitlines()]
    order = np.argsort(-np.array(scores)) # to reverse sort
    
    # TODO: do using multiple top patches and use NMS
    
    # get the top matches from each and intersection
    matches = readMatches(matchesdir, i, [order[0]])
    scores = computeScores(matches, i-1, imgslist)
    allscores += np.array(scores)
    for match in matches[:20]:
      fout.write('%d:%f ' % (match[1], match[0]))
    fout.write('\n')
  fout.close()
  print allscores / len(testlist)

def readMatches(matchesdir, i, boxids):
  fpath = os.path.join(matchesdir, str(i) + '.txt')
  lines = readLines(fpath, boxids)
  matches = []
  assert(len(boxids) == 1) # for now
  for line in lines:
    for el in line.strip().split():
      el2 = el.split(':')
      matches.append((float(el2[1]), int(el2[0])))
  return matches

def computeScores(matches, imgid, imgslist):
  # remove the exact match
  sameornot = [getImgId(m[1]) == imgid for m in matches]
  del matches[np.where(sameornot)[0][0]]

  clses = [getClass(m[1], imgslist) for m in matches]
  cls = getClass(imgid * 10000 + 1, imgslist)
  hits = [c == cls for c in clses]
  scores = []
  for i in [1,3,5,10,20]: # for mP
    scores.append(float(sum(hits[:i])) / i)
  scores.append(sum(hits[:3]) > 0) # for atleast 1/3
  scores.append(sum(hits[:10]) > 0)
  return scores

def readLines(fpath, lnos): # lnos must be 0 indexed
  order = np.argsort(np.array(lnos))
  lines = []
  with open(fpath) as f:
    for i, line in enumerate(f):
      if i in lnos:
        lines.append(line)
  return list(np.array(lines)[order])

def getClass(idx, imgslist):
  imid = idx / 10000
  return os.path.dirname(imgslist[imid])

def getImgId(idx):
  return idx / 10000

if __name__ == '__main__':
  main()

