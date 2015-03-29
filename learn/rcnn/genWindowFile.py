import os, sys, math
from PIL import Image

seldir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes'
scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores'
trainlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList_120.txt'
imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus'
outfpath = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/cnn/001_rcnn/window_file.txt'


def readListFromFile(fpath, dtype):
  f = open(fpath)
  elts = [dtype(el) for el in f.read().splitlines()]
  return elts

def computeImageSize(fpath):
  im = Image.open(fpath)
  return im.size

def main():
  fout = open(outfpath, 'w')
  trainlist = readListFromFile(trainlistfile, int)
  imgslist = readListFromFile(imgslistfile, str)
  for i in range(len(trainlist)):
    fout.write('# %d\n' % i)
    idx = trainlist[i]
    fout.write('%s\n' % imgslist[idx - 1])
    fout.write('3\n')
    w,h = computeImageSize(os.path.join(imgsdir, imgslist[idx - 1]))
    fout.write('%d\n%d\n' % (h, w))
    selbox_str = readListFromFile(os.path.join(seldir, str(idx) + '.txt'), str)
    scores = readListFromFile(os.path.join(scoresdir, str(idx) + '.txt'), float)
    fout.write('%d\n' % len(scores))
    for j in range(len(scores)):
      fout.write('%f ' % scores[j])
      if scores[j] < 0.9:
        ov = 0
      else:
        ov = 1
      fout.write('%f ' % ov)
      selbox = [float(el) for el in selbox_str[j].split(',')]
      fout.write('%d %d %d %d\n' % (math.ceil(selbox[1] - 1), math.ceil(selbox[0] - 1),
            math.floor(selbox[3] - 1), math.floor(selbox[2] - 1)))
    
  fout.close()

if __name__ == '__main__':
  main()

