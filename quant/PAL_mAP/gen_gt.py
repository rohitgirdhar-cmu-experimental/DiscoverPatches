import glob
import os

def readList(fpath):
  f = open(fpath)
  lst = f.read().splitlines()
  f.close()
  return lst

def readName(fpath):
  return readList(fpath)[0].split(' ')[0][5:]

def writeOut(fpath, lst):
  try:
    os.makedirs(os.path.dirname(fpath))
  except:
    pass
  f = open(fpath, 'w')
  f.write('\n'.join(lst))
  f.close()

if 0:
  gtdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/mAP_eval/gtfiles/'
  imlistpath = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  outfpath = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/mAP_eval/QueryList.txt'
  testNdxesFile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
elif 1:
  gtdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/mAP_eval_train/gtfiles/'
  imlistpath = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  outfpath = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/mAP_eval_train/QueryList.txt'
  testNdxesFile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'


imlist = readList(imlistpath)
testlist = [int(el) for el in readList(testNdxesFile)]

fout = open(outfpath, 'w')
for id in testlist:
  imname = imlist[id - 1][:-4]
  cname = os.path.dirname(imname)
  good = [el[:-4] for el in imlist if el.startswith(cname)]
  junk = []
  ok = []
  writeOut(os.path.join(gtdir, imname + '_good.txt'), good)
  writeOut(os.path.join(gtdir, imname + '_ok.txt'), ok)
  writeOut(os.path.join(gtdir, imname + '_junk.txt'), junk)
  fout.write('%s %s %d\n' % (imname, imname, id))
fout.close()

