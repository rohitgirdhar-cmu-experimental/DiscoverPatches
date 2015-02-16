function getTopPatches()
imgsdir = 'dataset/PeopleAtLandmarks/corpus/';
outdir = 'results/Gradient_wts_txt/';
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
  I = rgb2gray(I);
  G = cat(3, imgradient(I, 'Sobel'), imgradient(I, 'Prewitt'), ...
      imgradient(I, 'CentralDifference'), imgradient(I, 'IntermediateDifference'), ...
      imgradient(I, 'Roberts'));
  
  boxes = dlmread(fullfile(boxesdir, [num2str(i) '.txt']));
  boxes = boxes(:, [2 1 4 3]);
  scores = zeros(size(boxes, 1), 5);
  for j = 1 : size(boxes, 1)
    subG = G(int32(ceil(boxes(j, 2))) : int32(floor(boxes(j, 4))), ...
        int32(ceil(boxes(j, 1))) : int32(floor(boxes(j, 3))), :);
    el = mean(mean(subG, 1), 2);
    scores(j, :) = reshape(el, 1, [], 1);
  end
  dlmwrite(fullfile(outdir, [num2str(i) '.txt']), scores, ' ');
end

