if 0
  modelspathdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/features/ESVM/testPeopleOnly/';
  outdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/scores_heatmap/esvm/testPeopleOnly/';
  testidxfpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTest.txt';
  imgsdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/corpus/';
  imgslistfpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/Images.txt';
  visdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/scores_heatmap/esvm/testPeopleOnly_vis/';
else
  modelspathdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/features/ESVM/trainPeopleOnly/';
  outdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/scores_heatmap/esvm/trainPeopleOnly/';
  testidxfpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTrain.txt';
  imgsdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/corpus/';
  imgslistfpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/Images.txt';
  visdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/scores_heatmap/esvm/trainPeopleOnly_vis/';
end

unix(['mkdir -p ' outdir]);
unix(['mkdir -p ' visdir]);

addpath('../../'); % for WtPicture
vis = 0;

f = fopen(imgslistfpath);
imgslist = textscan(f, '%s');
imgslist = imgslist{1};
fclose(f);

testidx = dlmread(testidxfpath);

for tid = testidx(:)'
  I = imread(fullfile(imgsdir, imgslist{tid}));
  try
    load(fullfile(modelspathdir, [num2str(tid) '.mat']));
    hmap = WtPicture(model{1}.model.w);
    hmap = imresize(hmap, [size(I, 1) size(I, 2)]);
  catch
    fprintf(2, 'Unable to read %d\n', tid);
    hmap = ones(size(I, 1), size(I, 2));
  end
  dlmwrite(fullfile(outdir, [num2str(tid) '.txt']), hmap);
  if vis
    imwrite(uint8(bsxfun(@times, double(I), hmap / max(hmap(:)))), fullfile(visdir, [num2str(tid) '.jpg']));
  end
end

