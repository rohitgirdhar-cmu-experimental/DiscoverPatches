function test()
if 0
  FEAT = 'pool5';
  featdir = ['~/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_text'];
  outdir = ['~/work/data/001_PALAnd1KHayesDistractor/query_scores/' FEAT];
  imgslistfile = '~/work/data/001_PALAnd1KHayesDistractor/ImgsList.txt';
else
  FEAT = 'fc7_PeopleOnly';
  featdir = ['~/work/data/002_ExtendedPAL/features/CNN/CNN_' FEAT];
  outdir = ['~/work/data/002_ExtendedPAL/query_scores/' FEAT];
  imgslistfile = '~/work/data/002_ExtendedPAL/lists/Images.txt';
  modelpath = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/models/model_' FEAT '.mat'];
  testimgidspath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTest.txt';
end

testimgids = readList2(testimgidspath);

addpath('bin/');
load(modelpath, 'llm');
% read images list
f = fopen(imgslistfile, 'r');
imgslist = textscan(f, '%s\n');
fclose(f);
imgslist = imgslist{1};

for i = testimgids(:)'
  im = imgslist{i};
  feats = dlmread(fullfile(featdir, strrep(im, '.jpg', '.txt')));
  feats = bsxfun(@minus, feats, llm.mu);
  feats = bsxfun(@rdivide, feats, llm.sigma);
  ntest = size(feats, 1);
  scores = liblinear_predict(zeros(ntest, 1), sparse(feats), llm);
  dlmwrite(fullfile(outdir, [num2str(i) '.txt']), scores);
  fprintf('Done for %d\n', i);
end

function lst = readList2(fpath)
f = fopen(fpath);
lst = textscan(f, '%d');
lst = lst{1};
fclose(f);

