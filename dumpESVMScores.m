function getTopPatches()
imgsdir = 'dataset/PeopleAtLandmarks/corpus/';
masksdir = '/IUS/vmr105/rohytg/projects/003_SelfieSeg/009_BgMatch/dataset/PeopleAtLandmarks/masks/';
wtsdir = 'results/PeopleAtLandmarks_ESVMModels/';
outdir = 'results/ESVM_wts_txt/';
boxesdir = 'results/selsearch_boxes';
f = fopen(fullfile(imgsdir, '../', 'ImgsList.txt'));
imgslist = textscan(f, '%s\n');
imgslist = imgslist{1};
fclose(f);

i = 0;
for img = imgslist(:)'
  i = i + 1;
  img = img{:};

  I = imread(fullfile(imgsdir, img));
  load(fullfile(wtsdir, strrep(img, 'jpg', 'mat')), 'model');
  W = imresize(WtPicture(model{1}.model.w), [size(I, 1) size(I, 2)]);

  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']));
  boxes = boxes(:, [2 1 4 3]);
  scores = zeros(size(boxes, 1), 1);
  for j = 1 : size(boxes, 1)
    subW = W(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))));
    scores(j) = mean(subW(:));
  end
  dlmwrite(fullfile(outdir, [num2str(i) '.txt']), scores);
end

