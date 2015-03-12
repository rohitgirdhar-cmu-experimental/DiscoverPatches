function visualizeTopScorers_dump()
% dump top matches into a human readable text file to easily 
% generate webpages using PyHTMLWriter
%scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/query_scores_rbf_10K/';
scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/all_query_scores/query_scores_gt/';
outdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/learn_good_patches/scratch/top_patches_text';
boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes';
imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt';
imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus';
matchesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_refined';
detail = 1;

f = fopen(imgslistfile);
imgslist = textscan(f, '%s');
imgslist = imgslist{1};
fclose(f);

for i = 1 : 1 : 120
  thisoutfpath = fullfile(outdir, [num2str(i) '.txt']);

  if exist(thisoutfpath, 'file') || exist([thisoutfpath '.lock'], 'dir')
    continue;
  end
  unix(['mkdir -p ' thisoutfpath '.lock']);

  fprintf('Doing for %d\n', i);
  scores = dlmread(fullfile(scoresdir, [num2str(i) '.txt']));
  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']), ',');
  boxes = boxes(:, [2 1 4 3]);
  [scores, order] = sort(scores, 'descend');
  boxes = boxes(order, :);
  fout = fopen(thisoutfpath, 'w');
  for j = 1 : min(20, size(boxes, 1))
    if scores(j) < 0.01
      break
    end
    fprintf(fout, '%s; %f,%f,%f,%f; %f; ', imgslist{i}, boxes(j, 1), boxes(j, 2), boxes(j, 3) - boxes(j, 1), boxes(j, 4) - boxes(j, 2), scores(j));
    if detail
      line = getLine(fullfile(matchesdir, [num2str(i) '.txt']), order(j));
      matches = cellfun(@(x) strsplit(x, ':'), strsplit(line, ' '), 'UniformOutput', false);
      marked = {};
      for m = 1 : min(numel(matches), 20)
        mt = matches{m};
        [imname,box] = getImgBBox(mt{1}, imgsdir, imgslist, boxesdir);
        fprintf(fout, '%s:%f,%f,%f,%f:%s ', imname, box(1), box(2), box(3), box(4), mt{2});
      end
    end
    fprintf(fout, '\n');
  end
  fclose(fout);
  unix(['rmdir ' thisoutfpath '.lock']);
end

function out = getLine(fpath, lno)
fid = fopen(fpath);
out = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);
out = out{1}{lno};

function [imname, box] = getImgBBox(idx, imgsdir, imgslist, bboxdir)
idx = str2num(idx);
img = int32(floor(idx / 10000));
id = mod(idx, 10000);
imname = imgslist{img + 1};
line = getLine(fullfile(bboxdir, [num2str(img + 1) '.txt']), id);
box = strsplit(line, ',');
box = cellfun(@(x) str2num(x), box);
box = box(:, [2 1 4 3]);
% return bbox in [x,y,w,h] format
box = [box(1) box(2) box(3) - box(1) box(4) - box(2)];


function out = getMarkedImg(idx, imgsdir, imgslist, boxesdir)
idx = str2num(idx);
img = int32(floor(idx / 10000));
id = mod(idx, 10000);
I = imread(fullfile(imgsdir, imgslist{img + 1}));
line = getLine(fullfile(boxesdir, [num2str(img + 1) '.txt']), id);
box = strsplit(line, ',');
box = cellfun(@(x) str2num(x), box);
box = box(:, [2 1 4 3]);
out = insertRect(I, box);

