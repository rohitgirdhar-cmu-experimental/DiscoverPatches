import glob
import os

def readList(fpath):
  f = open(fpath)
  lst = f.read().splitlines()
  f.close()
  return lst

def readName(fpath):
  return readList(fpath)[0].split(' ')[0][5:]

query_files = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/gtfiles/'
imlistpath = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
outfpath = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/QueryList.txt'

imlist = readList(imlistpath)

qfiles = glob.glob(os.path.join(query_files, '*_query.txt'))
fout = open(outfpath, 'w')
for qfile in qfiles:
  qfname = qfile[:-10]
  qname = readName(qfile)
  try:
    qidx = imlist.index(qname + '.jpg') + 1
  except:
    qidx = -1
  fout.write('%s %s %d\n' % (os.path.basename(qfname), qname, qidx))
fout.close()

