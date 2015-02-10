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
  imwrite(M, fullfile(thisoutdir, 'mask.jpg'));

  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']));
  boxes = boxes(:, [2 1 4 3]);
  scores = zeros(size(boxes, 1), 1);
  for j = 1 : size(boxes, 1)
    subW = W(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))));
    subM = M(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))));
    if sum(subM(:) > 0) / (size(subM, 1) * size(subM, 2)) < 0.1
      scores(j) = mean(subW(:));
    end
  end

  [scores, order] = sort(scores, 'descend');
  boxes = boxes(order, :);
  %% store the blocks with > 0 wt, for debug
  if 0
    for j = 1 : size(boxes, 1)
      if scores(j) > 0
        figure(1);
        imshow(insertRect(I, boxes(j, :)));
        waitforbuttonpress();
      end
    end
  end

  mI2 = I;
  for j = 1 : 500
    mI2 =  insertRect(mI2, boxes(j, :));
  end
  imwrite(imresize(mI2, 0.5), fullfile(thisoutdir, 'bbox.jpg'));

  [boxes, pick] = ...
    esvm_nms(boxes(1 : 500, :), 0.85);
  
  scores_nms = scores(pick);
  [~, order_nms] = sort(scores_nms, 'descend');
  boxes = boxes(order_nms, :);
  if 0 % see NMS output
    mI2 = I;
    for j = 1 : size(boxes, 1)
      mI2 =  insertRect(mI2, boxes(j, :));
    end
    imshow(mI2);
    waitforbuttonpress();
  end

%  [~, order_pick] = sort(scores(pick), 'descend');
  for j = 1 : size(boxes, 1)
    %markedI = insertRect(markedI, ...
    %    [boxes(order(j), 1) boxes(order(j), 2) ...
    %      boxes(order(j), 3), boxes(order(j), 4)]);
    markedI = insertRect(I, boxes(j, :), [], [], 0.3);
    imwrite(imresize(markedI, 0.5), fullfile(thisoutdir, [num2str(j) '.jpg']));
  end
  disp(['Done for ' num2str(i)]);
end

