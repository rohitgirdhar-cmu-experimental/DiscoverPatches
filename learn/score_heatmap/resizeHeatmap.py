import numpy as np
import os, Image
import scipy.misc
heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/full'
outdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/small/'
visdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/vis/'
imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus/'
imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
vis = True

with open(imgslistfile) as f:
  imgslist = f.read().splitlines()

for i in range(1, 120):
  hmap_orig = np.loadtxt(os.path.join(heatmapdir, str(i) + '.txt'));
  hmap = scipy.misc.imresize(hmap_orig, (40, 40))
  np.savetxt(os.path.join(outdir, str(i) + '.txt'), hmap, fmt='%0.4f');

  if vis:
    I = scipy.misc.imread(os.path.join(imgsdir, imgslist[i - 1]), flatten=1)
    sz = (hmap_orig.shape[0], hmap_orig.shape[1])
    hmap = (hmap_orig * 1.0 / np.max(hmap_orig)) * 255.0
    R = 0.2 * np.dstack((I,I,I)) + 0.8 * np.dstack((hmap, np.zeros(sz), np.zeros(sz)))
    R = R.astype(np.uint8)
    R = Image.fromarray(R)
    R.save(os.path.join(visdir, str(i) + '.jpg'))
