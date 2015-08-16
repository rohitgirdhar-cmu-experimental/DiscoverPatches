function test()
if 0
  FEAT = 'pool5';
  featdir = ['~/work/data/001_PALAnd1KHayesDistractor/features/CNN_' FEAT '_text'];
  outdir = ['~/work/data/001_PALAnd1KHayesDistractor/query_scores/' FEAT];
  imgslistfile = '~/work/data/001_PALAnd1KHayesDistractor/ImgsList.txt';
elseif 0
  FEAT = 'fc7_PeopleOnly';
  featdir = ['~/work/data/002_ExtendedPAL/features/CNN/CNN_' FEAT];
  outdir = ['~/work/data/002_ExtendedPAL/query_scores/' FEAT];
  imgslistfile = '~/work/data/002_ExtendedPAL/lists/Images.txt';
  modelpath = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/models/model_' FEAT '.mat'];
  testimgidspath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTest.txt';
elseif 0
  FEAT = 'fc7_TrainOnly';
  featdir = ['~/work/data/003_HussianHotels/features/CNN_' FEAT];
  outdir = ['~/work/data/003_HussianHotels/query_scores/' FEAT];
  imgslistfile = '~/work/data/003_HussianHotels/lists/Images.txt';
  modelpath = ['/IUS/homes4/rohytg/work/data/003_HussianHotels/models/model_' FEAT '.mat'];
  testimgidspath = '/IUS/homes4/rohytg/work/data/003_HussianHotels/lists/NdxesTest.txt';
elseif 0
  FEAT = 'fc7';
  featdir = ['~/work/data/004_OxBuildings/features/CNN_' FEAT];
  outdir = ['~/work/data/004_OxBuildings/query_scores/' FEAT];
  imgslistfile = '~/work/data/004_OxBuildings/lists/Images.txt';
  modelpath = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/models/model_' FEAT '_PeopleOnly.mat'];
  testimgidspath = '/IUS/homes4/rohytg/work/data/004_OxBuildings/lists/NdxesTest.txt';
elseif 0
  FEAT = 'fc7_test_hdf5';
  featdir = ['~/work/data/002_ExtendedPAL/features/CNN/CNN_' FEAT];
  outdir = ['~/work/data/002_ExtendedPAL/query_scores/' FEAT];
  imgslistfile = '~/work/data/002_ExtendedPAL/lists/Images.txt';
  modelpath = ['/IUS/homes4/rohytg/work/data/002_ExtendedPAL/models/model_fc7_PeopleOnly.mat'];
  testimgidspath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTest.txt';
  feat_file_type = 'hdf5';
elseif 1
  featdir = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/features/CNN/fc7_test/'];
%  outdir = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/query_scores/CNN/test/'];
  outdir = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/query_scores/Jegou13/test/'];
  imgslistfile = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/Images.txt';
%  modelpath = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/models/model_fc7_cnnScoring.mat'];
  modelpath = ['/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/models/model_fc7_Jegou13Scoring.mat'];
  testimgidspath = '/IUS/vmr105/rohytg/data/005_ExtendedPAL2_moreTest/lists/NdxesPeopleTest.txt';
  feat_file_type = 'hdf5';
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
  if feat_file_type == 'hdf5'
    feats = h5read(fullfile(featdir, strrep(im, '.jpg', '.h5')), '/feats')';
  else
    feats = dlmread(fullfile(featdir, strrep(im, '.jpg', '.txt')));
  end
  feats = bsxfun(@minus, feats, full(llm.mu));
  feats = bsxfun(@rdivide, feats, full(llm.sigma));
  ntest = size(feats, 1);
  scores = liblinear_predict(zeros(ntest, 1), sparse(double(feats)), llm);
  dlmwrite(fullfile(outdir, [num2str(i) '.txt']), scores);
  fprintf('Done for %d\n', i);
  clear feats;
end

function lst = readList2(fpath)
f = fopen(fpath);
lst = textscan(f, '%d');
lst = lst{1};
fclose(f);

