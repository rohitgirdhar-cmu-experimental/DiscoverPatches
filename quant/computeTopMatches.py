#!/usr/bin/python -tt

# computes top matches for every image, using a given a scoring

import os, sys, math, subprocess, random
import numpy as np

if 1:
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
  method = 'svr_linear_10000'
  #method = 'gt'
  scoresdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/' + method
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals/' + method + '.txt.random'
else:
  # for full img matching case
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches_refined/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches_top.txt'

takeTopN = -1 # -n = random n patches
              # -1 = 1 random patch
              # 1 = top match
              # 5 = top 5 matches
MAXBOXPERIMG = 10000

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]
  
  fout = open(outfpath, 'w')
  allscores = np.zeros((1, 7))
  for i in testlist:
    try:
      with open(os.path.join(scoresdir, str(i) + '.txt')) as f:
        patchscores = [float(el) for el in f.read().splitlines()]
    except:
      patchscores = [1] # assuming only 1 patch in the image and with score = 1
    order = np.argsort(-np.array(patchscores)) # to reverse sort
    
    # TODO: do using multiple top patches and use NMS
    
    selected = []
    if takeTopN < 0:
      selected = random.sample(range(len(order)), -takeTopN)
    elif takeTopN > 0:
      selected = list(order[:takeTopN])

    # get the top matches from each and intersection
    matches = readMatches(matchesdir, i, selected)
    scores = computeScores(matches, i-1, imgslist)
    allscores += np.array(scores)
    fout.write('%d; ' % ((i-1) * MAXBOXPERIMG + order[0])) # query box
    for match in matches[:20]:
      fout.write('%d:%f ' % (match[1], match[0]))
    fout.write('\n')
  fout.close()
  print 'mp1,mp3,mp5,mp10,mp20,atleast1/3,atleast1/10'
  print ','.join([str(s) for s in list(allscores / len(testlist))[0]])

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

