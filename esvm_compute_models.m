function esvm_compute_models(imgsdir, masksdir, negdir, outdir)
addpath(genpath('~/projects/001_ESVM/ESVM/esvm_matching_vanilla/'));
if ~exist('negdir', 'var')
  negdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/corpus_mods/esvm/trainNneg/'
end
if ~exist('masksdir', 'var')
%  masksdir = '/IUS/vmr105/rohytg/projects/003_SelfieSeg/009_BgMatch/dataset/PeopleAtLandmarks/masks'
end
if ~exist('imgsdir', 'var')
  imgsdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/corpus_mods/esvm/test/'
end
if ~exist('outdir', 'var')
  outdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/features/ESVM/test/'
end
all_imgs = getAllFiles(imgsdir);

for img = all_imgs(:)'
  [path, imid, ~] = fileparts(img{:});
  thisoutdir = fullfile(outdir, path);
  if ~exist(thisoutdir, 'dir')
    unix(['mkdir -p ' thisoutdir]);
  end
  outfpath = fullfile(thisoutdir, [imid, '.mat']);
  lockoutfpath = [outfpath '.lock']
  if exist(outfpath, 'file') || exist(lockoutfpath, 'file')
    disp(['Already done for ' img{:}]);
    continue;
  end
  mkdir(lockoutfpath);

  I = imread(fullfile(imgsdir, img{:}));
  %M = im2bw(imread(fullfile(masksdir, img{:})));
  disp(['Read ' img{:} '. Computing model...']);
  model = esvm_train_single_exemplar(I, [1 1 size(I, 2) size(I, 1)], negdir);
%      'mask_img', M); % not masking for now
  save(outfpath, 'model');

  unix(['rmdir ' lockoutfpath]);
  close all;
  clearvars -except negdir masksdir outdir imgsdir;
end

