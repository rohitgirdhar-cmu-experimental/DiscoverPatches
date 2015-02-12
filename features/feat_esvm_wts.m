function feat_esvm_wts()
addpath('../');
imgsdir = '../dataset/PeopleAtLandmarks/corpus/';
masksdir = '/IUS/vmr105/rohytg/projects/003_SelfieSeg/009_BgMatch/dataset/PeopleAtLandmarks/masks/';
wtsdir = '../results/PeopleAtLandmarks_ESVMModels/';
outdir = '../results/features/esvm_wts';
boxesdir = '../results/selsearch_boxes';
if ~exist(outdir, 'dir')
  mkdir(outdir);
end
f = fopen(fullfile(imgsdir, '../', 'ImgsList.txt'));
imgslist = textscan(f, '%s\n');
imgslist = imgslist{1};
fclose(f);

try
    matlabpool open 8
catch
end

parfor i = 1 : numel(imgslist)
  img = imgslist{i};
%  thisoutdir = fullfile(outdir, strrep(img, '.jpg', ''));
%  if ~exist(thisoutdir, 'dir')
%    unix(['mkdir -p ' thisoutdir]);
%  end

  I = imread(fullfile(imgsdir, img));
  M = imresize(im2bw(imread(fullfile(masksdir, img))), [size(I, 1) size(I, 2)]);
  model = loadModel(fullfile(wtsdir, strrep(img, 'jpg', 'mat')));
  W = imresize(WtPicture(model{1}.model.w), [size(I, 1) size(I, 2)]);
  
%  wtdI = bsxfun(@times, double(I), W);
%  imwrite(wtdI, fullfile(thisoutdir, 'wtd.jpg'));
%  imwrite(M, fullfile(thisoutdir, 'mask.jpg'));

  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']));
  boxes = boxes(:, [2 1 4 3]);
  scores = zeros(size(boxes, 1), 1);
  for j = 1 : size(boxes, 1)
    subW = W(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))));
    scores(j) = mean(subW(:));
%    if you want to keep foreground scores = 0 artificially
%    subM = M(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
%        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))));
%    if sum(subM(:) > 0) / (size(subM, 1) * size(subM, 2)) < 0.1
%      scores(j) = mean(subW(:));
%    end
  end


  if 0 % DEBUG
    [scores, order] = sort(scores, 'descend');
    boxes = boxes(order, :);
    for j = 1 : size(boxes, 1)
      if scores(j) > 0
        figure(1);
        imshow(insertRect(I, boxes(j, :)));
        waitforbuttonpress();
      end
    end
  end
  
  % save out the wts in mat file in outdir
  saveScores(fullfile(outdir, [num2str(i) '.mat']), scores);
end

function [model] = loadModel(path)
model = [];
load(path, 'model');

function saveScores(path, scores)
save(path, 'scores', '-v7.3');
