function []=label2dataset_ball_neg_png(input_png_dir,labelname,output_negdir)
maxw=640;
maxh=480;
if ~exist(output_negdir,'dir')
    mkdir(output_negdir);
end

%% input all images init
all_part_names = dir([input_png_dir,'/*.png']);
all_names = {};
all_imgs = {};
for i=1:length(all_part_names)
    all_names{i} = all_part_names(i).name;
    
    imname = [input_png_dir,'/',all_names{i}];
    y_gray = imread(imname); 
    all_imgs{i} = y_gray;
end
imgname2img_map = containers.Map(all_names,all_imgs);


%% get labels
finp = fopen(labelname,'r');
lli=1;
label_imnames = {};
rects = [];
while true
    tline = fgetl(finp);
    if ~ischar(tline)
        break;
    end
    pos = strfind(tline,'.png');
    
    label_imnames{lli} = tline(1:pos+3);
    rects(lli,:) = str2num(tline(pos+4:end));
    lli=lli+1;
end
rects = rects+1;

%% cut negs
for i=1:length(label_imnames)
    imname = label_imnames{i};
    rect = rects(i,:);
    
    [~,a,postfix] = fileparts(imname);
    img = [];
    try
        img = imgname2img_map([a,postfix]);
    catch
        fprintf('\n%s\n',imname);
        disp('error');
        return;
    end   
    
    trect = double(rect);
    mid = [(trect(1)+trect(3))/2, (trect(2)+trect(4))/2];
    bottom_length = max(abs(trect(3)-trect(1)),abs(trect(4)-trect(2)));
    txx = round(mid(1)-bottom_length):round(mid(1)+bottom_length);
    tyy = round(mid(2)-bottom_length):round(mid(2)+bottom_length);
    txx = txx(txx>0);
    tyy = tyy(tyy>0);
    txx = txx(txx<=maxw);
    try
        tyy = tyy(tyy<=maxh);
    catch
        fprintf('hehe\n');
    end

    img(tyy,txx,:) = round(255*rand(length(tyy),length(txx),3));  
    imgname2img_map([a,postfix]) = img;
%     imshow(yneg);

end

%% write negative img
allKeys = keys(imgname2img_map);
for i=1:length(allKeys)
    imname = allKeys{i};
    [~,fn,~]=fileparts(imname);
    write_neg_name = [output_negdir,'/',fn,'.png'];
    yneg = imgname2img_map(allKeys{i});
    imwrite(yneg,write_neg_name);
end



fclose(finp);
end
