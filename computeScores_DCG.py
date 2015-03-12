#!/usr/bin/python

import os, sys, math, subprocess

matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList.txt'
boxesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/'
outdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores'

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]

  for i in testlist:
    print i
    outfile = open(os.path.join(outdir, str(i) + '.txt'), 'w')
    basecls,_,_ = getClassImgId((i - 1) * 10000 + 1, imgslist)
    with open(os.path.join(matchesdir, str(i) + '.txt')) as f:
      for line in f.read().splitlines():
        rel = []
        matches = line.split()
        for match in matches[0 : 21]:
          idx, score = match.split(':')
          idx = int(idx)
          score = float(score)
          cls,imid,featid = getClassImgId(idx, imgslist)
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
  return (os.path.dirname(lst[el / 10000]), el / 10000 + 1, el % 10000)

if __name__ == '__main__':
  main()

