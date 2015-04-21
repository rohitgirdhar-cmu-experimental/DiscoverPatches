function visualizeRetrievals()
addpath('..'); % for insertRect
%method = 'rbf_10K';
method = 'gt';
if 0
  scoresdir = ['/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/query_scores_' , method];
  retfpath = ['/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals/'  method  '.txt'];
  outdir = ['/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals_vis/'  method  '/'];
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes';
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus';
elseif 0
  retfpath = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/matches_top/test.txt';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/scratch/vis/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/selsearch_boxes';
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/Images.txt';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/lists/NdxesPeopleTest.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0006_ExtendedPAL/corpus';
elseif 1
  retfpath = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/matches_top/test.txt';
  outdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scratch/vis/';
  boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/selsearch_boxes';
  imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/lists/Images.txt';
  testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/scratch/good.txt';
  imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0007_HussianHotels/corpus';
end

f = fopen(imgslistfile);
imgslist = textscan(f, '%s');
imgslist = imgslist{1};
fclose(f);

f = fopen(retfpath);
retline = textscan(f, '%s', 'Delimiter', '\n');
retline = retline{1};
fclose(f);

f = fopen(testlistfile);
testlist = textscan(f, '%d');
testlist = testlist{1};
fclose(f);

for i = 1 : numel(testlist)
  outdpath = fullfile(outdir, num2str(testlist(i)));
  unix(['mkdir -p ' outdpath]);
  line = retline{testlist(i)};
  temp = strsplit(line, ';');
  q = temp{1}; rets = temp{2};
  qimg = getMarkedImg(q, imgsdir, imgslist, boxesdir);
  imwrite(qimg, fullfile(outdpath, 'q.jpg'));
  
  matches = strsplit(strtrim(rets), ' ');
  for j = 1 : min(numel(matches), 6)
    mel = strsplit(matches{j}, ':');
    mel = strsplit(mel{3}, ',');
    mid = mel{1};
    mimg = getMarkedImg(mid, imgsdir, imgslist, boxesdir);
    imwrite(mimg, fullfile(outdpath, [num2str(j) '.jpg']));
  end
  fprintf('Done for %d\n', testlist(i));
  fflush(stdout);
end

function out = getLine(fpath, lno)
fid = fopen(fpath);
out = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);
out = out{1}{lno};

function out = getMarkedImg(idx, imgsdir, imgslist, boxesdir) % idx must have 0 indexed img id
idx = str2num(idx);
img = int32(floor(idx / 10000));
id = mod(idx, 10000);
I = imread(fullfile(imgsdir, imgslist{img}));
line = getLine(fullfile(boxesdir, [num2str(img) '.txt']), id);
box = strsplit(line, ',');
box = cellfun(@(x) str2num(x), box);
box = box(:, [2 1 4 3]);
out = insertRect(I, box, [], 5);

function I = smallImg(I)
%I = imresize(I, [512, NaN]); % need newer octave.. too hard to compile! :(

