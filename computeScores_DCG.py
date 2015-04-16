#!/usr/bin/python

import os, sys, math, subprocess
from itertools import izip
from PIL import Image

if 0:
  datadir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus/'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList.txt'
  boxesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/'
  outdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores'
  MIN_DIM_BOX = 0.05 # set score for any box smaller than MIN_DIM_BOX*IM_WID x MIN_DIM_BOX*IM_HT to 0
else:
  datadir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/corpus/'
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/train/'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
  boxesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/selsearch_boxes/'
  outdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_scores'
  MIN_DIM_BOX = 0.05 # set score for any box smaller than MIN_DIM_BOX*IM_WID x MIN_DIM_BOX*IM_HT to 0
 

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]

  for i in testlist:
    print i
    outfile = open(os.path.join(outdir, str(i) + '.txt'), 'w')
    basecls,baseimname,_ = getClassImgId((i - 1) * 10000 + 1, imgslist)
    [im_wd, im_ht] = getImDims(imgslist[baseimname - 1])
    with open(os.path.join(matchesdir, str(i) + '.txt')) as f, open(os.path.join(boxesdir, str(i) + '.txt')) as f2:
      for line, line2 in izip(f, f2):
        if len(line.strip()) == 0:
          outfile.write('-1\n')
          continue
        box = [float(el) for el in line2.strip().split(',')]
        if box[2] - box[0] <= MIN_DIM_BOX * im_ht or box[3] - box[1] <= MIN_DIM_BOX * im_wd:
          outfile.write('0\n')
          continue
        rel = []
        matches = line.strip().split()
        for match in matches[0 : 21]:
          idx, score = match.split(':')
          idx = int(idx)
          score = float(score)
          cls,imid,_ = getClassImgId(idx, imgslist)
          if imid == i:
            continue # ignore the exact match, I want other images to match
          # Using refined results, so all matched images must be distinct
          if cls == basecls:
            rel.append(0.95 + 0.05 * score)
            #rel.append(1)
          else:
            rel.append(0)
        outfile.write('%f\n' % computeDCG(rel, 10)) # typically use top 10, but works with less too

def computeDCG(relevance, k):
  dcg = 0
  if k > len(relevance):
    print('WARNING: len(rel)(%d) < k(%d)' % (len(relevance), k))
  for i in range(min(k, len(relevance))):
    dcg += (math.pow(2.0, relevance[i]) - 1) / math.log(i + 2, 2) # since i is 0 indexed
  return dcg

def getClassImgId(el, lst):
  # old system (0,1)
  #return (os.path.dirname(lst[el / 10000]), el / 10000 + 1, el % 10000)
  # new system (1,1)
  return (os.path.dirname(lst[el / 10000 - 1]), el / 10000, el % 10000 - 1)

def getImDims(impath):
  im = Image.open(os.path.join(datadir, impath))
  return im.size

if __name__ == '__main__':
  main()

