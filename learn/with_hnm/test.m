FEAT = 'fc7';
featdir = ['~/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_text'];
outdir = ['~/work/data/001_PALAnd1KHayesDistractor/query_scores/' FEAT];
imgslistfile = '~/work/data/001_PALAnd1KHayesDistractor/ImgsList.txt';

addpath('bin/');
load(['model_' FEAT '.mat'], 'llm');
% read images list
f = fopen(imgslistfile, 'r');
imgslist = textscan(f, '%s\n');
fclose(f);
imgslist = imgslist{1};

for i = 121 : 237
  im = imgslist{i};
  feats = dlmread(fullfile(featdir, strrep(im, '.jpg', '.txt')));
  feats = bsxfun(@minus, feats, llm.mu);
  feats = bsxfun(@rdivide, feats, llm.sigma);
  ntest = size(feats, 1);
  scores = liblinear_predict(zeros(ntest, 1), sparse(feats), llm);
  dlmwrite(fullfile(outdir, [num2str(i) '.txt']), scores);
  fprintf('Done for %d\n', i);
end


