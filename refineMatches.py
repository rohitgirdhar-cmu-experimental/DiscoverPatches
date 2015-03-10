#!/usr/bin/python

# refines the matches file by removing matches in same image

import os, sys, math, subprocess
sys.path.append('learn/')
import locker

matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList.txt'
boxesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/'
outdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined'

def main():
  # read images list
  with open(imgslistpath) as f:
    imgslist = f.read().splitlines()
  with open(testlistpath) as f:
    testlist = [int(t) for t in f.read().splitlines()]
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  for i in testlist:
    outfpath = os.path.join(outdir, str(i) + '.txt')
    if not locker.lock(outfpath):
      continue
    print('Doing for %d' % i)
    outfile = open(outfpath, 'w')
    with open(os.path.join(matchesdir, str(i) + '.txt')) as f:
      for line in f.read().splitlines():
        matches = line.split()
        selected_imid = []
        for match in matches:
          idx, score = match.split(':')
          idx = int(idx)
          score = float(score)
          try:
            cls,imid,featid = getClassImgId(idx, imgslist)
          except:
            import pdb
            pdb.set_trace()
          if imid in selected_imid:
            continue
          outfile.write('%d:%f ' % (idx, score))
          selected_imid.append(imid)
        outfile.write('\n') 
    locker.unlock(outfpath)        

def getClassImgId(el, lst):
  return (os.path.dirname(lst[el / 10000]), el / 10000 + 1, el % 10000)

if __name__ == '__main__':
  main()

