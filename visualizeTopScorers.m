function visualizeTopScorers()
scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/matches_scores';
visdir = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/matches_visualization_temp';
boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/selsearch_boxes';
imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/ImgsList.txt';
imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/corpus';
matchesdir = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/matches';

f = fopen(imgslistfile);
imgslist = textscan(f, '%s');
imgslist = imgslist{1};
fclose(f);

for i = 1 : 2 : 205
  scores = dlmread(fullfile(scoresdir, [num2str(i) '.txt']));
  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']), ',');
  i
  boxes = boxes(:, [2 1 4 3]);
  [scores, order] = sort(scores, 'descend');
  boxes = boxes(order(1:100), :);
%  boxes = esvm_nms(boxes, 0.5);
  I = imread(fullfile(imgsdir, imgslist{i}));
  for j = 1 : min(10, size(boxes, 1))
    if scores(j) < 0.25
      break
    end
    J = insertRect(I, boxes(j, :));
    line = getLine(fullfile(matchesdir, [num2str(i) '.txt']), order(j));
    matches = cellfun(@(x) strsplit(x, ':'), strsplit(line, ' '), 'UniformOutput', false);
    marked = {};
    for m = 1 : numel(matches)
      mt = matches{m};
      marked{m} = getMarkedImg(mt{1}, imgsdir, imgslist, boxesdir);
    end
    thisoutdir = fullfile(visdir, num2str(i), num2str(j));
    unix(['mkdir -p ' thisoutdir]);
    imwrite(I, fullfile(thisoutdir, 'q.jpg'));
    for m = 1 : numel(matches)
      imwrite(marked{m}, fullfile(thisoutdir, [num2str(m) '.jpg']));
    end
    fprintf('Done for %d:%d\n', i, j);
  end
end

function out = getLine(fpath, lno)
fid = fopen(fpath);
out = textscan(fid, '%s', 'delimiter', '\n');
fclose(fid);
out = out{1}{lno};

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

