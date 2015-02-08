function getTopPatches()
imgsdir = 'dataset/PeopleAtLandmarks/corpus/';
masksdir = '/IUS/vmr105/rohytg/projects/003_SelfieSeg/009_BgMatch/dataset/PeopleAtLandmarks/masks/';
wtsdir = 'results/PeopleAtLandmarks_ESVMModels/';
outdir = 'results/top_windows/';
boxesdir = 'results/selsearch_boxes';
f = fopen(fullfile(imgsdir, '../', 'ImgsList.txt'));
imgslist = textscan(f, '%s\n');
imgslist = imgslist{1};
fclose(f);

i = 0;
for img = imgslist(:)'
  i = i + 1;
  img = img{:};
  thisoutdir = fullfile(outdir, strrep(img, '.jpg', ''));
  if ~exist(thisoutdir, 'dir')
    unix(['mkdir -p ' thisoutdir]);
  end

  I = imread(fullfile(imgsdir, img));
  M = imresize(im2bw(imread(fullfile(masksdir, img))), [size(I, 1) size(I, 2)]);
  load(fullfile(wtsdir, strrep(img, 'jpg', 'mat')), 'model');
  W = imresize(WtPicture(model{1}.model.w), [size(I, 1) size(I, 2)]);
  
  wtdI = bsxfun(@times, double(I), W);
  imwrite(wtdI, fullfile(thisoutdir, 'wtd.jpg'));

  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']));
  boxes = boxes(:, [2 1 4 3]);
  scores = zeros(size(boxes, 1), 1);
  for j = 1 : size(boxes, 1)
    subW = W(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))));
    subM = M(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))));
    if sum(subM(:) > 0) / (size(subM, 1) * size(subM, 2)) < 0.2
      scores(j) = mean(subW(:));
    end
  end
  [~, order] = sort(scores, 'descend');
  markedI = I;
  for j = 1 : 300
    markedI = insertRect(markedI, ...
        [boxes(order(j), 1) boxes(order(j), 2) ...
          boxes(order(j), 3), boxes(order(j), 4)]);
  end
  imwrite(imresize(markedI, 0.5), fullfile(thisoutdir, 'marked.jpg'));
end

