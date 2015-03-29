function convertDataToMat()
imgslistfile = '~/work/data/001_PALAnd1KHayesDistractor/ImgsList.txt';
txtfeatdir = '~/work/data/001_PALAnd1KHayesDistractor/features/CNN_fc7_text';
outdir = '~/work/data/001_PALAnd1KHayesDistractor/features/CNN_fc7_mats';

imgslist = readList(imgslistfile);
try
  matlabpool open 12;
catch
end

parfor i = 1 : 120 % only the training set
  im = fullfile(txtfeatdir, strrep(imgslist{i}, '.jpg', '.txt'));
  outfpath = fullfile(outdir, [num2str(i) '.mat']);
  if exist(outfpath, 'file') || exist([outfpath '.lock'], 'dir')
    continue
  end
  unix(['mkdir -p ' outfpath '.lock']);
  feats = dlmread(im);
  writeOut(outfpath, feats);
  unix(['rmdir ' outfpath '.lock']);
end

function lst = readList(fpath)
f = fopen(fpath);
lst = textscan(f, '%s');
lst = lst{1};
fclose(f);

function writeOut(outfpath, feats)
save(outfpath, 'feats', '-v7.3');
