function train()
% read all data
feat_file_type = 'mat'; % or could be h5
feat_file_naming = 'indexed'; % or could be imname
addpath('bin/');
if 0
  FEAT = 'pool5';
  FEATDIM = 9216;
  RANDSAMPLE = 1000;
  addpath('bin/');
  %featdir = ['/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_mats'];
  featdir = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/features/CNN_' FEAT '_mats'];
  scoresdir = '/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/matches_scores';
  modelfpath = ['model_' FEAT '.mat'];
elseif 0
  FEAT = 'fc7_PeopleOnly';
  FEATDIM = 4096;
  RANDSAMPLE = 1000;
  addpath('bin/');
  %featdir = ['/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_mats'];
  featdir = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/features/CNN/CNN_' FEAT '_mats'];
  scoresdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/matches_scores';
  modelfpath = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/models/model_' FEAT '.mat'];
  trainNdxesFpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTrain.txt';
elseif 0
  FEAT = 'fc7_TrainOnly';
  FEATDIM = 4096;
  RANDSAMPLE = 400;
  addpath('bin/');
  %featdir = ['/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_mats'];
  featdir = ['/IUS/homes4/rohytg/work/data/003_HussianHotels/features/CNN_' FEAT '_mats'];
  scoresdir = '/IUS/homes4/rohytg/work/data/003_HussianHotels/matches_scores/train/';
  modelfpath = ['/IUS/homes4/rohytg/work/data/003_HussianHotels/models/model_' FEAT '.mat'];
  trainNdxesFpath = '/IUS/homes4/rohytg/work/data/003_HussianHotels/lists/NdxesPeopleTrain+.txt';
elseif 0
  feat_file_type = 'h5';
  feat_file_naming = 'imname';
  FEAT = 'fc7';
  FEATDIM = 4096;
  RANDSAMPLE = 5000;
  %featdir = ['/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_mats'];
  featdir = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/features/CNN/fc7_train/'];
  scoresdir = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/matches_scores/CNN/train/';
  modelfpath = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/models/model_fc7_cnnScoring.mat'];
  trainNdxesFpath = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/NdxesPeopleTrain.txt';
  imgslistfpath = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/Images.txt';
elseif 1
  feat_file_type = 'h5';
  feat_file_naming = 'imname';
  FEAT = 'fc7';
  FEATDIM = 4096;
  RANDSAMPLE = 5000;
  %featdir = ['/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_mats'];
  featdir = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/features/CNN/fc7_train/'];
  scoresdir = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/matches_scores/Jegou13/train/';
  modelfpath = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/models/model_fc7_Jegou13Scoring.mat'];
  trainNdxesFpath = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/NdxesPeopleTrain.txt';
  imgslistfpath = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/Images.txt';
end

if exist('imgslistfpath', 'var')
  imgslist = readList(imgslistfpath, '%s');
end

trainNdxes = readList(trainNdxesFpath, '%d');

allfeats = sparse(0, FEATDIM);
allscores = zeros(0, 1);
for i = trainNdxes(:)'
  if strcmp(feat_file_type, 'mat')
    fext = '.mat';
  else
    fext = '.h5';
  end
  if strcmp(feat_file_naming, 'indexed')
    featFpath = fullfile(featdir, [num2str(i) fext]);
  else
    featFpath = fullfile(featdir, strrep(imgslist{i}, '.jpg', fext));
  end
  clear feats;
  try
    if strcmp(feat_file_type, 'mat')
      load(featFpath, 'feats'); % by default it stores each features as a column
    else
      feats = h5read(featFpath, '/feats');
      feats = double(feats');
    end
    scores = dlmread(fullfile(scoresdir, [num2str(i) '.txt']));
  catch
    fprintf(2, 'Unable to read %s\n', featFpath);
    continue
  end
  rem = (scores == -1); % remove those which don't have assoc scores
  feats(rem, :) = [];
  scores(rem, :) = [];
  if RANDSAMPLE ~= -1
    nsel = min(RANDSAMPLE, size(feats, 1));
    sel = randsample(size(feats, 1), nsel);
    feats = feats(sel, :);
    scores = scores(sel, :);
  end
  allfeats(end + 1 : end + size(feats, 1), :) = ...
    sparse(feats);
  allscores(end + 1 : end + size(scores, 1), :) = scores;
  fprintf('Read for %d\n', i);
end

fprintf('Total %d features\n', size(allfeats, 1));

mu = mean(allfeats);
sigma = std(allfeats);
allfeats = bsxfun(@minus, allfeats, mu);
allfeats = bsxfun(@rdivide, allfeats, sigma);

ll_opts = '-s 11 -B 10';

llm = liblinear_train(allscores, allfeats, ll_opts);
llm.mu = mu;
llm.sigma = sigma;
save(modelfpath, 'llm');

function lst = readList(fpath, formatstr)
f = fopen(fpath);
lst = textscan(f, formatstr);
lst = lst{1};
fclose(f);

