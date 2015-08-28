import numpy as np
import os, Image
import scipy.misc

delim = ' '
if 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/full'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractors/split/TrainList.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/query/full'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/query/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/query/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/train/full'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/train/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/train/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/crossval/full'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/crossval/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/crossval/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/human/full'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/human/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/human/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  vis = True
  delim = ','
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/david/Saliency/full'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/david/Saliency/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/david/Saliency/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt'
  vis = True
  delim = ','
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/train/full'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/train/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/train/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/NdxesTrain.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/query/full/'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/query/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/query/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/NdxesPeopleTest.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/scores_heatmap/query/full/'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/scores_heatmap/query/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/scores_heatmap/query/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/corpus/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/test/full/'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/test/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/test/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/corpus_resized/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesPeopleTest.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/Jegou13/train/full/'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/Jegou13/train/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/Jegou13/train/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/corpus_resized/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesPeopleTrain.txt'
  vis = True
elif 0:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/train/full/'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/train/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/train/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/corpus_resized/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesPeopleTrain.txt'
  vis = True
elif 1:
  heatmapdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/crossval_train_n10/full/'
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/crossval_train_n10/small/'
  visdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/scores_heatmap/CNN/crossval_train_n10/vis/'
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/corpus_resized/'
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/Images.txt'
  testimgidxsfile = '/srv2/rgirdhar/Work/Datasets/processed/0010_ExtendedPAL_moreTest/lists/NdxesPeopleTrain.txt'
  vis = True

with open(imgslistfile) as f:
  imgslist = f.read().splitlines()
testimgidxs = [int(el) for el in open(testimgidxsfile).read().splitlines()]

for i in testimgidxs:
  hmap_orig = np.loadtxt(os.path.join(heatmapdir, str(i) + '.txt'), delimiter=delim);
  hmap = scipy.misc.imresize(hmap_orig, (40, 40))
  np.savetxt(os.path.join(outdir, str(i) + '.txt'), hmap, fmt='%0.4f');

  if vis:
    I = scipy.misc.imread(os.path.join(imgsdir, imgslist[i - 1]), flatten=1)
    sz = (hmap_orig.shape[0], hmap_orig.shape[1])
    hmap = (hmap_orig * 1.0 / np.max(hmap_orig)) * 255.0
    try:
      R = 0.2 * np.dstack((I,I,I)) + 0.8 * np.dstack((hmap, np.zeros(sz), np.zeros(sz)))
    except Exception, e:
      print e
      print 'Continuing'
      continue
    R = R.astype(np.uint8)
    R = Image.fromarray(R)
    R.save(os.path.join(visdir, str(i) + '.jpg'))
