#!/usr/bin/python

import os, sys, math, subprocess

matchesdir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/matches'
imgslistpath = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/ImgsList.txt'
testlistpath = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/split/TestList.txt'
boxesdir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/selsearch_boxes/'
outdir = '/home/rgirdhar/data/Work/Datasets/processed/0001_PALn1KDistractor/matches_scores'

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
    basefeatid = 0
    with open(os.path.join(matchesdir, str(i) + '.txt')) as f:
      for line in f.read().splitlines():
        basefeatid += 1
        if not line:
          hitratio = -1
        else:
          matches = line.split()
          hits = 0
          pos = 0
          total_possible_score = 0
#          selboxes = []
          selected_imid = []
          for match in matches[0 : 30]:
            score_for_this_pos = 1.0 / (1.02 ** pos)
            idx, score = match.split(':')
            idx = int(idx)
            score = float(score)
#            box = getBox(idx, imgslist)
            if score < 0.7:
              break
            cls,imid,featid = getClassImgId(idx, imgslist)
            if imid == i:
              continue # no points for getting exact duplicate or another patch from same image
                       # usually ends up giving me perturbations of the same patch
#            if selboxes and max(computeOverlaps(selboxes, (imid, box))) > 0.8: # too expensive!!!
#              continue # A close by box from this image has already been selected
            if imid in selected_imid: # no points for matching to another patch in same image
              continue
            if cls == basecls:
              hits += score_for_this_pos * score # also depend on how good the match is
              selected_imid.append(imid)
#              selboxes.append((imid, box))
            pos += 1
            total_possible_score += score_for_this_pos
          hitratio = hits / (total_possible_score + 0.000001)
        outfile.write('%f\n' % hitratio)

def getClassImgId(el, lst):
  return (os.path.dirname(lst[el / 10000]), el / 10000 + 1, el % 10000)

def getBox(idx, lst):
  _, imid, featid = getClassImgId(idx, lst)
  boxfpath = os.path.join(boxesdir, str(imid) + '.txt')
  fp = open(boxfpath)
  out = ''
  for i, line in enumerate(fp):
    if i == featid - 1:
      out = line
      break
  fp.close()
  return [float(el) for el in out.split(',')]

def computeOverlaps(selboxes, box):
  # selboxes : <image id, box dim>
  # box: img id, box dim (in selsearch format)
  overlaps = []
  for sbox in selboxes:
    if sbox[0] != box[0]:
      overlaps.append(0)
    else:
      overlaps.append(computeBoxOverlap(sbox[1], box[1]))
  return overlaps

def computeBoxOverlap(b1, b2):
  # convert to a sane format
  b1 = convertSane(b1)
  b2 = convertSane(b2)
  b1 = [str(el) for el in b1]
  b2 = [str(el) for el in b2]
  out = subprocess.check_output('./rectInt ' + b1[0] + ' ' + b1[1] + ' ' + b1[2] + ' ' + b1[3] + ' '
      + b2[0] + ' ' + b2[1] + ' ' + b2[2] + ' ' + b2[3], shell=True)
  return float(out)

def convertSane(b):
  return [b[1], b[0], b[3] - b[1], b[2] - b[0]]

if __name__ == '__main__':
  main()

