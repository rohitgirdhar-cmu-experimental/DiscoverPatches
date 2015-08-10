import pdb
import os

imfile = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
gtdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/gtfiles/'
gtdir_ndxes = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/gtfiles_ndxes/'
testfile = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
queryfile = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/QueryList.txt'

def readList(fpath):
  f = open(fpath)
  res = f.read().splitlines()
  f.close()
  return res

def convertToIdx(lst, imlist):
  res = []
  for el in lst:
    res.append(imlist.index(el + '.jpg') + 1)
  return res

def writeOut(lst, fpath):
  f = open(fpath, 'w')
  for el in lst:
    f.write('%d\n' % el)
  f.close()

imlist = open(imfile).read().splitlines()
testlist = [int(el) for el in open(testfile).read().splitlines()]
qlist = [el.split() for el in open(queryfile).read().splitlines()]
qlist = [(el[0], el[1], int(el[2])) for el in qlist]

for qel in qlist:
  goodpath = os.path.join(gtdir, qel[0] + '_good.txt')
  okpath = os.path.join(gtdir, qel[0] + '_ok.txt')
  junkpath = os.path.join(gtdir, qel[0] + '_junk.txt')

  goodlist = readList(goodpath);
  oklist = readList(okpath);
  junklist = readList(junkpath);

  goodlist = convertToIdx(goodlist, imlist)
  oklist = convertToIdx(oklist, imlist)
  junklist = convertToIdx(junklist, imlist)

  outgoodpath = os.path.join(gtdir_ndxes, str(qel[2]) + '_good.txt')
  outokpath = os.path.join(gtdir_ndxes, str(qel[2]) + '_ok.txt')
  outjunkpath = os.path.join(gtdir_ndxes, str(qel[2]) + '_junk.txt')

  writeOut(goodlist, outgoodpath)
  writeOut(oklist, outokpath)
  writeOut(junklist, outjunkpath)


