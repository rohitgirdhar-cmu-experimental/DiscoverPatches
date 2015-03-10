function I = insertRect(I, box, col, lwid, dim)
% box is xmin ymin xmax ymax
origI = I;
box = [ceil(box(1)) ceil(box(2)) floor(box(3)) floor(box(4))];
if ~exist('col', 'var') || isempty(col)
  col = [255, 0, 0];
end
if ~exist('lwid', 'var') || isempty(lwid)
  lwid = 3;
end
if ~exist('dim', 'var') || isempty(dim)
  dim = 0;
end
col = reshape(col, 1, 1, 3);
try
  I(box(:, 2) : box(:, 4), box(:, 1) : box(:, 1) + lwid - 1, :) = ...
    repmat(col, box(:, 4) - box(:, 2) + 1, lwid);
catch
end
try
  I(box(:, 2) : box(:, 4), box(:, 3) : box(:, 3) + lwid - 1, :) = ...
    repmat(col, box(:, 4) - box(:, 2) + 1, lwid);
catch
end
try
  I(box(:, 2) : box(:, 2) + lwid - 1, box(:, 1) : box(:, 3), :) = ...
    repmat(col, lwid, box(:, 3) - box(:, 1) + 1);
catch
end
try
  I(box(:, 4) : box(:, 4) + lwid - 1, box(:, 1) : box(:, 3), :) = ...
    repmat(col, lwid, box(:, 3) - box(:, 1) + 1);
catch
end

% a messy fix in case the above leads to a changed image size
I = I(1:size(origI, 1), 1:size(origI, 2), :);

if dim
  I(1 : box(:, 2), :, :) = uint8(origI(1 : box(:, 2), :, :) .* 0.5);
  I(box(:, 4) + lwid - 1 : end, :, :) = uint8(origI(box(:, 4) + lwid - 1 : end, :, :) .* 0.5);
  I(:, 1 : box(:, 1), :) = uint8(origI(:, 1 : box(:, 1), :) .* 0.5);
  I(:, box(:, 3) + lwid - 1: end, :) = uint8(origI(:, box(:, 3) + lwid - 1 : end, :) .* 0.5);
end
