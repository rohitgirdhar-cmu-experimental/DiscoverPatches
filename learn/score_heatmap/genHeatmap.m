function genHeatmap()

if 0
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/';
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores/';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/full/';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList_120.txt';
elseif 0
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/selsearch_boxes/';
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/query_scores/fc7_PeopleOnly/';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/query/full/';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt';
elseif 0
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/selsearch_boxes/';
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/matches_scores/';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/train/full/';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt';
elseif 1
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/selsearch_boxes/';
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/learn/train_crossval_scores/';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scores_heatmap/crossval/full/';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTrain.txt';
elseif 0
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/Images.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/corpus/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/selsearch_boxes/';
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/matches_scores/train/';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/train/full/';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/NdxesTrain.txt';
elseif 0
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/Images.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/corpus/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/selsearch_boxes/';
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/query_scores/fc7_TrainOnly/';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scores_heatmap/query/full/';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/NdxesPeopleTest.txt';
elseif 0
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/Images.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/corpus/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/selsearch_boxes/';
  scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/query_scores/fc7/';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/scores_heatmap/query/full/';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0008_OxBuildings/lists/NdxesTest.txt';
end


imgslist = readImgsList(imgslistfile);
trainidxs = readNumList(testlistfile);
for i = trainidxs(:)'
  i
  fflush(stdout);
  im = imgslist{i};
  [w,h] = getImgSize(fullfile(imgsdir, im));
  hmap = zeros(h, w);
  boxes = readBoxes(fullfile(boxesdir, [num2str(i) '.txt']));
  scores = readNumList(fullfile(scoresdir, [num2str(i) '.txt']));
  for j = 1 : size(scores, 1)
    if scores(j) < 0
      continue
    end
    hmap(boxes(j, 2) : boxes(j, 4), boxes(j, 1) : boxes(j, 3)) += scores(j);
  end
  % save the map and a visualization
  dlmwrite(fullfile(outdir, [num2str(i) '.txt']), hmap, 'delimiter', ' ', 'precision', '%.4f');
end

function lst = readImgsList(fpath)
f = fopen(fpath, 'r');
lst = textscan(f, '%s');
fclose(f);
lst = lst{1};

function scores = readNumList(fpath)
scores = dlmread(fpath);

function [wid, ht] = getImgSize(fpath)
s = imfinfo(fpath);
wid = s.Width;
ht = s.Height;

function boxes = readBoxes(fpath)
boxes = dlmread(fpath);
boxes = [ceil(boxes(:, 2)) ceil(boxes(:, 1)) ...
  floor(boxes(:, 4)) floor(boxes(:, 3))];

function res = rgb2gray(I)
R = I(:, :, 1);
G = I(:, :, 2);
B = I(:, :, 3);
res = 0.2989 * R + 0.5870 * G + 0.1140 * B;

