#!/usr/bin/python -tt

# computes top matches for every image, using a given a scoring

import os, sys, math, subprocess, random
import numpy as np
sys.path.append('../')
from computeScores_DCG import computeDCG
from nms import non_max_suppression_fast

takeTopN = 1 # -n = random n patches
              # -1 = 1 random patch
              # 1 = top match
              # 5 = top 5 matches
select = 1 # 0=> select nth. 1=> select 1..nth (only valid for top matches, not random)
nmsTh = 0.9 # set = -1 for no NMS
          # else, set a threshold between [0, 1]

if 0:
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined/'
  selboxdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
#  method = 'svr_rbf_10000'
#  method = 'gt'
  method = 'svr_linear_FullData_liblinear'
#  method = 'svr_linear_FullData_liblinear_pool5'
#  method = 'deep_regressor_5K'
  scoresdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/' + method
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals/' + method + '__' + str(takeTopN) + '__nms' + str(nmsTh) + '.txt'
elif 1:
  # for full img matching case
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/fullImg/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/fullImg.txt'
  nmsTh = -1 # set = -1 for no NMS
elif 0:
  # for patch case
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/test/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_top/test.txt'
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/query_scores/fc7_PeopleOnly/'
  nmsTh = -1 # set = -1 for no NMS
else:
  # for full img matching case
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches_refined/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt'
  outfpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches_top.txt'

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
    try:
      with open(os.path.join(scoresdir, str(i) + '.txt')) as f:
        patchscores = [float(el) for el in f.read().splitlines()]
    except:
      patchscores = [1] # assuming only 1 patch in the image and with score = 1
    order = np.argsort(-np.array(patchscores)) # to reverse sort
  
    if nmsTh >= 0:
      order = performNMS(order, os.path.join(selboxdir, str(i) + '.txt'), nmsTh)
    
    selected = []
    if takeTopN < 0:
      selected = random.sample(range(len(order)), -takeTopN)
    elif takeTopN > 0 and select == 1:
      selected = list(order[:takeTopN])
    elif takeTopN > 0 and select == 0:
      selected = [order[takeTopN - 1]]

    # get the top matches from each and intersection
    matches = readMatches(matchesdir, i, selected)
    scores = computeScores(matches, i, imgslist)
    allscores += np.array(scores)
    # also add these scores to class specific lists as well
    testcls = getClass(i - 1, imgslist)
    if testcls in allscores_cls.keys():
      allscores_cls[testcls][0] += np.array(scores)
      allscores_cls[testcls][1] += 1
    else:
      allscores_cls[testcls] = [np.array(scores), 1]
    
    ndist = countMatchesOfClass(matches, imgslist, 10, 'NaturalImages')
    if testcls in numdists_cls.keys():
      numdists_cls[testcls][0] += ndist
      numdists_cls[testcls][1] += 1
    else:
      numdists_cls[testcls] = [ndist, 1]
    
    qboxes = [(i) * MAXBOXPERIMG + el + 1 for el in selected]
    fout.write('%s; ' % ','.join([str(el) for el in qboxes])) # query box
    for match in matches[:20]:
      fout.write('%d:%f:%s ' % (match[1], match[0], 
            ','.join([str(el) for el in match[2]])))
    fout.write('\n')
  fout.close()
  print 'mp1,mp3,mp5,mp10,mp20,atleast1/3,atleast1/10,DCG/10'
  print ','.join([str(s) for s in list(allscores / len(testlist))[0]])
  # print for each class
  classes = allscores_cls.keys()
  print ','.join(classes)
  dcg_scores = [allscores_cls[cls][0][7] / allscores_cls[cls][1] for cls in classes]
  print ','.join([str(el) for el in dcg_scores])
  print ','.join([str(numdists_cls[cls][0] * 1.0 / numdists_cls[cls][1]) 
      for cls in classes])

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
def computeScores(matches, imgid, imgslist):
  # remove the exact match
  sameornot = [m[1] == imgid for m in matches]
  del matches[np.where(sameornot)[0][0]]

  clses = [getClass(m[1], imgslist) for m in matches]
  cls = getClass(imgid, imgslist)
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
  clses = [getClass(m[1], imgslist) for m in matches][:n]
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

if __name__ == '__main__':
  main()

