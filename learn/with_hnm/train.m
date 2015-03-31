function train()
% read all data
FEAT = 'pool5';
FEATDIM = 9216;
RANDSAMPLE = 2000;
addpath('bin/');
featdir = ['/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_mats'];
scoresdir = '/IUS/homes4/rohytg/work/data/001_PALAnd1KHayesDistractor/matches_scores';
modelfpath = ['model_' FEAT '.mat'];

allfeats = sparse(0, FEATDIM);
allscores = zeros(0, 1);
for i = 1 : 120
  load(fullfile(featdir, [num2str(i) '.mat']), 'feats');
  scores = dlmread(fullfile(scoresdir, [num2str(i) '.txt']));
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

mu = mean(allfeats);
sigma = std(allfeats);
allfeats = bsxfun(@minus, allfeats, mu);
allfeats = bsxfun(@rdivide, allfeats, sigma);

ll_opts = '-s 11 -B 10';

llm = liblinear_train(allscores, allfeats, ll_opts);
llm.mu = mu;
llm.sigma = sigma;
save(modelfpath, 'llm');

