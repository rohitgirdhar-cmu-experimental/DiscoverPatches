function convertDataToMat()
if 0
  imgslistfile = '~/work/data/001_PALAnd1KHayesDistractor/ImgsList.txt';
  txtfeatdir = '~/work/data/001_PALAnd1KHayesDistractor/features/CNN_pool5_text';
  outdir = '~/work/data/001_PALAnd1KHayesDistractor/features/CNN_pool5_mats';
else
  imgslistfile = '~/work/data/002_ExtendedPAL/lists/Images.txt';
  txtfeatdir = '~/work/data/002_ExtendedPAL/features/CNN/CNN_fc7_PeopleOnly/';
  outdir = '~/work/data/002_ExtendedPAL/features/CNN/CNN_fc7_PeopleOnly_mats';
  peopleIdxsFpath = '~/work/data/002_ExtendedPAL/lists/NdxesPeople.txt';
end

imgslist = readList(imgslistfile);
peopleIdxsList = readList2(peopleIdxsFpath);

for i = peopleIdxsList(:)' % only the training set
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

function lst = readList2(fpath)
f = fopen(fpath);
lst = textscan(f, '%d');
lst = lst{1};
fclose(f);

function writeOut(outfpath, feats)
save(outfpath, 'feats', '-v7.3');
