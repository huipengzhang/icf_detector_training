function [img_new] = adjust_contrast_onelayer(grayimg,newmean,newstd)

% replace the 'L' layer with the processed data and then convert
% the image back to the RGB colorspace
grayimg = double(grayimg);
im_size_origin = size(grayimg);
L = reshape(grayimg(:,:),[],1);
temp = sort(L);

L_mean = mean(temp);
L_mid = median(temp);
L_std = std(temp);
img_new_min = 0;
img_new_max = 255;

img_new = (L-L_mean)/L_std*newstd+newmean;
img_new(img_new<img_new_min)=img_new_min;
img_new(img_new>img_new_max)=img_new_max;
img_new = reshape(img_new,im_size_origin);

end