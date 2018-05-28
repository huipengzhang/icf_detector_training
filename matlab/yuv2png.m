function [] = yuv2png(inputdir,outputdir)
if ~exist(outputdir,'dir');
    mkdir(outputdir);
end

all_yuv_names = dir([inputdir,'/*.yuv']);
for i=1:length(all_yuv_names)
    name = [inputdir,'/',all_yuv_names(i).name];
    fin = fopen(name,'rb');
    p = fread(fin,640*480*2);
    y = p(1:2:length(p));
    imgy = reshape(y,640,[])';
    imgall = repmat(imgy,[1,1,3]);
    imgall = uint8(imgall);
    imshow(imgall);
    
    outputname = [outputdir,'/',all_yuv_names(i).name(1:end-3),'png'];
    imwrite(imgall,outputname);
    fclose(fin);
end



end