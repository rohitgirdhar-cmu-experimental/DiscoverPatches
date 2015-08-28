function train()
% read all data
addpath('bin/');
if 0
  FEAT = 'fc7_PeopleOnly';
  FEATDIM = 4096;
  featdir = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/features/CNN/CNN_' FEAT '_mats'];
  scoresdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/matches_scores';
  trainNdxesFpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTrain.txt';
  outcrossvalscores = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/learn/train_crossval_scores/';
  nfolds = 3;
else
  FEATDIM = 4096;
  featdir = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/features/CNN/fc7_train/';
  scoresdir = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/matches_scores/Jegou13/train/';
  trainNdxesFpath = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/NdxesPeopleTrain.txt';
  imgslistfpath = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/Images.txt';
  outcrossvalscores = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/learn/train_crossval_scores_n20/';
  nfolds = 20;
end

trainNdxes = readList(trainNdxesFpath, '%d');
imgslist = readList(imgslistfpath, '%s');
nTrainIdxs = numel(trainNdxes);

allfeats = {};
allscores = {};
for i = trainNdxes(:)'
  featFpath = fullfile(featdir, strrep(imgslist{i}, '.jpg', '.h5'));
  clear feats;
  try
    feats = double(h5read(featFpath, '/feats')');
    scores = dlmread(fullfile(scoresdir, [num2str(i) '.txt']));
  catch
    fprintf(2, 'Unable to read %s\n', featFpath);
    continue
  end
  %if RANDSAMPLE ~= -1
  %  nsel = min(RANDSAMPLE, size(feats, 1));
  %  sel = randsample(size(feats, 1), nsel);
  %  feats = feats(sel, :);
  %  scores = scores(sel, :);
  %end
  allfeats{i} = ...
    sparse(feats);
  allscores{i} = scores;
  fprintf('Read for %d\n', i);
end

% generate folds
numInFold = uint32(nTrainIdxs / nfolds);
folds = {};
for i = 1 : nfolds
  if i ~= nfolds
    folds{i} = trainNdxes((i - 1) * numInFold + 1 : i * numInFold);
  else
    folds{i} = trainNdxes((i - 1) * numInFold + 1 : end);
  end
end

for fold_i = 1 : numel(folds)
  idxs = zeros(0, 1);
  for fold_i2 = 1 : numel(folds)
    if fold_i2 == fold_i
      continue;
    end
    idxs(end + 1 : end + numel(folds{fold_i2}), :) = folds{fold_i2};
  end
  [thisfeats, thisscores] = getRelData(allfeats, allscores, idxs, FEATDIM);
  
  rem = (thisscores == -1); % remove those which don't have assoc scores
  thisfeats(rem, :) = [];
  thisscores(rem, :) = [];
  
  mu = mean(thisfeats);
  sigma = std(thisfeats);
  thisfeats = bsxfun(@minus, thisfeats, mu);
  thisfeats = bsxfun(@rdivide, thisfeats, sigma);

  ll_opts = '-s 11 -B 10';

  llm = liblinear_train(thisscores, thisfeats, ll_opts);
  
  % run the prediction on held out set
  testidxs = folds{fold_i};
  for testid = testidxs(:)'
    [testfeats, testscores] = getRelData(allfeats, allscores, [testid], FEATDIM);
    testfeats = bsxfun(@minus, testfeats, mu);
    testfeats = bsxfun(@rdivide, testfeats, sigma);
    predictedscores = liblinear_predict(testscores, sparse(testfeats), llm);
    writeOut(predictedscores, outcrossvalscores, testid);
  end
end

function lst = readList(fpath, formatstr)
f = fopen(fpath);
lst = textscan(f, formatstr);
lst = lst{1};
fclose(f);

function [thisfeats, thisscores] = getRelData(allfeats, allscores, idxs, FEATDIM)
thisfeats = sparse(0, FEATDIM);
thisscores = zeros(0, 1);
for el = 1 : numel(idxs)
  f1 = allfeats{idxs(el)};
  s1 = allscores{idxs(el)};
  thisfeats(end + 1 : end + size(f1, 1), :) = f1;
  thisscores(end + 1 : end + size(f1, 1), :) = s1;
end

function writeOut(predictedscores, outdir, testid)
outfpath = fullfile(outdir, [num2str(testid) '.txt']);
dlmwrite(outfpath, predictedscores(:), 'delimiter', '\n');

