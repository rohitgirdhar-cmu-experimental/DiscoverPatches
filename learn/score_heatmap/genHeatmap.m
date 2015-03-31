function genHeatmap()

match_scores = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores/';
imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt';
imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus/';
boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes/';
scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores/';
outdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/scores_heatmap/full/';
testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TrainList_120.txt';

imgslist = readImgsList(imgslistfile);
trainidxs = readNumList(testlistfile);
for i = trainidxs(:)'
  im = imgslist{i};
  [w,h] = getImgSize(fullfile(imgsdir, im));
  hmap = zeros(h, w);
  boxes = readBoxes(fullfile(boxesdir, [num2str(i) '.txt']));
  scores = readNumList(fullfile(scoresdir, [num2str(i) '.txt']));
  for j = 1 : size(scores, 1)
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

