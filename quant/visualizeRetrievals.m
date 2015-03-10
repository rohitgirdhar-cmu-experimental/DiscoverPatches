function visualizeRetrievals()
addpath('..'); % for insertRect
method = 'rbf_10K';
scoresdir = ['/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/query_scores_' , method];
retfpath = ['/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals/'  method  '.txt'];
outdir = ['/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/retrievals_vis/'  method  '/'];
boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes';
imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt';
testlistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/split/TestList.txt';
imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus';

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
  line = retline{i};
  temp = strsplit(line, ';');
  q = temp{1}; rets = temp{2};
  qimg = getMarkedImg(q, imgsdir, imgslist, boxesdir);
  imwrite(qimg, fullfile(outdpath, 'q.jpg'));
  
  matches = strsplit(strtrim(rets), ' ');
  for j = 1 : min(numel(matches), 20)
    mel = strsplit(matches{j}, ':');
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
I = imread(fullfile(imgsdir, imgslist{img + 1}));
line = getLine(fullfile(boxesdir, [num2str(img + 1) '.txt']), id);
box = strsplit(line, ',');
box = cellfun(@(x) str2num(x), box);
box = box(:, [2 1 4 3]);
out = insertRect(I, box);

function I = smallImg(I)
%I = imresize(I, [512, NaN]); % need newer octave.. too hard to compile! :(

