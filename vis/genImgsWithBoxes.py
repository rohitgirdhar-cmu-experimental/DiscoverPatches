imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/corpus_resized/'
imgslistfpath = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/Images.txt'
testlistfpath = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesTest.txt'
selsearchdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/features/selsearch_boxes/'
retfpath = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/matches_top/test_final_patch_1xfull.txt'
outdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/temp/001_Patch+FullFinalOutput'

import os
import numpy as np
import cv2
import subprocess
import math

def drawBoxes(I, boxes):
  for i in range(np.shape(boxes)[0]):
    cv2.rectangle(I, (int(math.ceil(boxes[i][1])), int(math.ceil(boxes[i][0]))), (int(math.floor(boxes[i][3])), int(math.floor(boxes[i][2]))), (0,0,255), 3)

with open(imgslistfpath) as f:
  imgslist = f.read().splitlines()
with open(retfpath) as f:
  retrievals = f.read().splitlines()
selboxes = {}
with open(testlistfpath) as f:
  for tid in f.read().splitlines():
    selboxes[tid] = np.loadtxt(os.path.join(selsearchdir, tid + '.txt'), delimiter=',')
print 'Read selboxes'

for retrieval in retrievals:
  qimg = retrieval.split(';')[0]
  boxids = [int(el) for el in qimg.split(',')]
  imid = boxids[0] / 10000
  allqboxes = []
  for boxid in boxids:
    allqboxes.append(selboxes[str(imid)][boxid % 10000 - 1, :])
  I = cv2.imread(os.path.join(imgsdir, imgslist[imid - 1]))
  drawBoxes(I, np.array(allqboxes))
  subprocess.call('mkdir -p ' + outdir + '/' + str(imid), shell=True)
  cv2.imwrite(os.path.join(outdir, str(imid), 'q.jpg'), I)
  rets = retrieval.split(';')[1].strip().split(' ')[:20]
  for ret in rets:
    mid = ret.split(':')[0]
    mboxes = [int(el) for el in ret.strip().split(':')[2].split(',')]
    allmboxes = []
    for boxid in mboxes:
      allmboxes.append(selboxes[mid][boxid % 10000 - 1, :])
    I = cv2.imread(os.path.join(imgsdir, imgslist[int(mid) - 1]))
    drawBoxes(I, np.array(allmboxes))
    cv2.imwrite(os.path.join(outdir, str(imid), str(mid) + '.jpg'), I) 

