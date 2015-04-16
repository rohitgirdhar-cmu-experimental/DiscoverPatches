#!/usr/bin/python

# refines the matches file by removing matches in same image

import os, sys, math, subprocess
sys.path.append('learn/')
import locker

if 0:
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/All.txt'
  outdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined'
elif 0:
  # patches, train
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches/train'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
  outdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/train/'
elif 1:
  # patches, test
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches/test'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/test/'
elif 0:
  # full img case
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches/fullImg'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  outdir = '/home/rgirdhar/data/Work/Datasets/processed/0006_ExtendedPAL/matches_refined/fullImg/'
else:
  # for the full img case
  matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches'
  imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/All.txt'
  outdir = '/home/rgirdhar/data/Work/Datasets/processed/0004_PALn1KHayesDistractor/aux_matches/matches_fullImg/matches_refined'


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
            cls,imid,_ = getClassImgId(idx, imgslist)
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
  # older way, with (0, 1) indexing
  # return (os.path.dirname(lst[el / 10000]), el / 10000 + 1, el % 10000)
  # newer, with (1,1) indexing
  return (os.path.dirname(lst[el / 10000 - 1]), el / 10000, el % 10000 - 1)

if __name__ == '__main__':
  main()

