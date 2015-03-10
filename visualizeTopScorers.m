function visualizeTopScorers()
scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_scores';
%scoresdir = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/query_scores';
visdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches_visualization';
%visdir = '/srv2/rgirdhar/Work/Datasets/processed/0001_PALn1KDistractor/query_visualization_temp';
boxesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/selsearch_boxes';
imgslistfile = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/ImgsList.txt';
imgsdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/corpus';
matchesdir = '/srv2/rgirdhar/Work/Datasets/processed/0004_PALn1KHayesDistractor/matches';
ST = 1;
detail = 0; % = 1 if want to store all the matching images too
detaillist = 11:40;
PRINTN = 30;

f = fopen(imgslistfile);
imgslist = textscan(f, '%s');
imgslist = imgslist{1};
fclose(f);

for i = ST : 1 : 237
  scores = dlmread(fullfile(scoresdir, [num2str(i) '.txt']));
  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']), ',');
  i
  boxes = boxes(:, [2 1 4 3]);
  [scores, order] = sort(scores, 'descend');
  boxes = boxes(order(1:100), :);
%  boxes = esvm_nms(boxes, 0.5);
  I = imread(fullfile(imgsdir, imgslist{i}));
  summaryI = I;
  unix(['mkdir -p ' fullfile(visdir, num2str(i))]);
  for j = 1 : min(20, size(boxes, 1))
    if scores(j) < 0.2
      break
    end
    J = insertRect(I, boxes(j, :));
    summaryI = insertRect(summaryI, boxes(j, :));
    thisoutdir = fullfile(visdir, num2str(i), num2str(j));
    if detail || (exist('detaillist', 'var') && any(detaillist == i))
      fprintf('Score: %f\n', scores(j))
      disp('Saving detail');
      fflush(stdout);
      line = getLine(fullfile(matchesdir, [num2str(i) '.txt']), order(j));
      matches = cellfun(@(x) strsplit(x, ':'), strsplit(line, ' '), 'UniformOutput', false);
      marked = {};
      for m = 1 : min(PRINTN, numel(matches))
        mt = matches{m};
        marked{m} = getMarkedImg(mt{1}, imgsdir, imgslist, boxesdir);
      end
      unix(['mkdir -p ' thisoutdir]);
      imwrite(smallImg(J), fullfile(thisoutdir, 'q.jpg'));
      for m = 1 : min(PRINTN, numel(matches))
        imwrite(smallImg(marked{m}), fullfile(thisoutdir, [num2str(m) '.jpg']));
      end
    end
    fprintf('Done for %d:%d\n', i, j);
    fflush(stdout);
  end
  imwrite(smallImg(summaryI), fullfile(visdir, num2str(i), 'summary.jpg'));
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

function I = smallImg(I)
%I = imresize(I, [512, NaN]); % need newer octave.. too hard to compile! :(

