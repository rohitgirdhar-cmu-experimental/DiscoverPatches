function im = WtPicture(w, bs)
% bs: block size in the image
if ~exist('bs','var')
  bs = 20;
end
% HOGpicture(w, bs)
% Make picture of positive HOG weights.

% make pictures of positive weights bs
s = size(w);
w(w < 0) = 0;
im = zeros(bs*s(1), bs*s(2));
for i = 1:s(1),
  iis = (i-1)*bs+1:i*bs;
  for j = 1:s(2),
    jjs = (j-1)*bs+1:j*bs;
    im(iis,jjs) = mean(w(i,j,:));
  end
end
