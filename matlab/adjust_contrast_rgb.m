function [imgun] = adjust_contrast_rgb(imgo,newmean,newstd)
% input rgb image, output rgb image


%% colorful image
% srgb2lab = makecform('srgb2lab');
% lab2srgb = makecform('lab2srgb');
% shadow_lab = applycform(imgo, srgb2lab); % convert to L*a*b*
%% gray image
shadow_lab = imgo;

% the values of luminosity can span a range from 0 to 100; scale them
% to [0 1] range (appropriate for MATLAB(R) intensity images of class double)
% before applying the three contrast enhancement techniques
max_luminosity = 255;
L = double(shadow_lab(:,:,1));
L = adjust_contrast_onelayer(L/max_luminosity, newmean, newstd);

shadow_imadjust = shadow_lab;
shadow_imadjust(:,:,1) = L*max_luminosity;

%% colorful image
% imgun = applycform(shadow_imadjust, lab2srgb);
%% gray image
imgun = repmat(shadow_imadjust(:,:,1),[1,1,3]);


imgun = uint8(imgun);

end