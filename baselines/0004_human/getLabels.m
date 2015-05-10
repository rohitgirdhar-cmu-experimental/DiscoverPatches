outdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/scores_heatmap/human/';
testidxfpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesPeopleTest.txt';
imgsdir = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/corpus/';
imgslistfpath = '/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/Images.txt';
unix(['mkdir -p ' outdir]);

f = fopen(imgslistfpath);
imgslist = textscan(f, '%s');
imgslist = imgslist{1};
fclose(f);

testidx = dlmread(testidxfpath);

testidx = testidx(randperm(numel(testidx)));

f = figure;
%testidx = [113,1480,1737,3110,3112];
for tid = testidx(:)'
  outfpath = fullfile(outdir, [num2str(tid) '.txt']);
  if exist(outfpath, 'file')
    continue;
  end
  tid
  I = imread(fullfile(imgsdir, imgslist{tid}));
  imshow(I);
  rect = getrect(f);
  T = zeros(size(I, 1), size(I, 2));
  T(rect(2) : rect(2) + rect(4), rect(1) : rect(1) + rect(3)) = 1;
  dlmwrite(outfpath, T);
end

