import os, sys, math, subprocess, random
import numpy as np
import h5py
sys.path.append('/srv2/rgirdhar/Work/Code/0003_DiscoverPatches/DiscoverPatches/')
from computeScores_DCG import computeDCG
from selectPatches import selectPatches

imgslistpath = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
simmatdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/pairwise_matches_bin/'
scoresdir_bin = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/train_crossval_scores_bin/'
trainidxs = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
matchesdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/matches/train/'
N = 5;
MAXBOXPERIMG = 10000

with open(imgslistpath) as f:
  imgslist = f.read().splitlines()
with open(trainidxs) as f:
  testlist = [int(t) for t in f.read().splitlines()]

def main():
  maxscore = -1
  maxscore_param = -1
  params =  np.arange(-5, 5, 0.2)
  tot = evalParamValue(params)
  print 'top: ', params[np.argmax(tot)]

def evalParamValue(params):
  tot = np.zeros((len(params), 1))
  for i in testlist:
    sims = readHDF5(os.path.join(simmatdir_bin, str(i) + '.h5'), 'sims')
    patchscores = readHDF5(os.path.join(scoresdir_bin, str(i) + '.h5'), 'scores')
    noMatchesList = getNoMatchesExistList(matchesdir, i) # returns 0 indexed

    pi = 0
    for param in params:
      selected,selscores = selectPatches(patchscores, sims, param, N, noMatchesList)

      # get the top matches from each and intersection
      matches = readMatches(matchesdir, i, selected)
      score = computeScoresDCG_wrapper(matches, i, imgslist)
      #score = computeScoresMP3_wrapper(matches, i, imgslist)
      tot[pi] += score # the DCG
      print '%d : %f : %f' % (i, param, score)
      pi += 1
  #return tot * 1.0 / len(testlist)
  return tot * 1.0 / len(testlist)

# outputs [(score, imid, imfeatids)...] // imid is not the imid*10K+featid
def readMatches(matchesdir, i, boxids):
  fpath = os.path.join(matchesdir, str(i) + '.txt')
  lines = readLines(fpath, boxids)
  allmatches = []
  for line in lines:
    matches = []
    for el in line.strip().split()[:50]:
      el2 = el.split(':')
      matches.append((float(el2[1]), int(el2[0])))
    allmatches.append(matches)
  matches = mergeRanklists(allmatches)
  return matches

# returns [(score, imgid, imfeatids)..]
def mergeRanklists(allmatches):
  imid2score = {}
  imid2feats = {} # store what bounding boxes in this image matched
  for matches in allmatches:
    for match in matches:
      imid = getImgId(match[1])
      if imid not in imid2score.keys():
        imid2score[imid] = match[0]
        imid2feats[imid] = [match[1]]
      else:
        imid2score[imid] += match[0]
        imid2feats[imid].append(match[1])
  res = imid2score.items()
  res = sorted(res, key=lambda tup: tup[1], reverse=True) # remember, reverse sort!
  res = [(m[1], m[0], imid2feats[m[0]]) for m in res]
  return res

# matches must be [(score, imid)...]
def computeScoresDCG_wrapper(matches, imgid, imgslist):
  # remove the exact match
  sameornot = [m[1] == imgid for m in matches]
  del matches[np.where(sameornot)[0][0]]

  clses = [getClass(m[1] - 1, imgslist) for m in matches]
  cls = getClass(imgid - 1, imgslist)
  hits = [c == cls for c in clses]
  return computeDCG(hits[:20], 20)

def computeScoresMP3_wrapper(matches, imgid, imgslist):
  # remove the exact match
  sameornot = [m[1] == imgid for m in matches]
  del matches[np.where(sameornot)[0][0]]

  clses = [getClass(m[1] - 1, imgslist) for m in matches]
  cls = getClass(imgid - 1, imgslist)
  hits = [c == cls for c in clses]
  return sum(hits[:3]) / 3.0

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

def getClass(imid, imgslist): # imgid here must be 0-indexed!!!
  return os.path.dirname(imgslist[imid])

def getImgId(idx):
  return idx / MAXBOXPERIMG

def readHDF5(fpath, dbname):
  f = h5py.File(fpath, 'r')
  data = f[dbname][:]
  f.close()
  return data

def getNoMatchesExistList(dpath, imgid):
  # imgid is 1 indexed
  fpath = os.path.join(dpath, str(imgid) + '.txt')
  f = open(fpath, 'r')
  lengths = [len(line) for line in f.read().splitlines()]
  f.close()
  return [i for i, e in enumerate(lengths) if e == 0]

if __name__ == '__main__':
  main()

