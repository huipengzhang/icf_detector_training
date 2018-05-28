function [] = images2unnormalized_images(input_img_dir,output_img_dir,new_mid_std)
if ~exist(output_img_dir,'dir')
    mkdir(output_img_dir);
end

if nargin<3    

% new_mid_std = [23,15;30,30;70,40;170,40;230,40];
new_mid_std = ...
    [...
%     23,15;...
%     30,30;...
    90,40;...
    100,40;...
    120,40;...
%     150,40;...
%     200,40;...
%     230,40...
    ];

end

% for j=1:100
%     fprintf('Processing %d dir.\n',j);
%     imgdin = sprintf([input_img_dir,'/%d'],j);    
%     imgdout = sprintf([output_img_dir,'/%d'],j);
    imgdin = input_img_dir;
    imgdout = output_img_dir;

    if ~exist(imgdout,'dir')
        mkdir(imgdout);
    end
    
    dd = dir([imgdin,'/*.png']);
    for i=1:length(dd)     
        imgnin = [imgdin,'/',dd(i).name];
        
        imgo = imread(imgnin);
        for k=1:size(new_mid_std,1)            
            imgnew = adjust_contrast_rgb(imgo,new_mid_std(k,1)/255,new_mid_std(k,2)/255);
%             imshow(imgnew);
            imgnout = sprintf([imgdout,'/',dd(i).name,'ext_%d_%d.png'],new_mid_std(k,1),new_mid_std(k,2));
            imwrite(imgnew,imgnout);            
        end
    end    
% end

end








