function []=label2dataset_ball_pos_png(labelname,output_posdir, resize_pos)
maxw=640;
maxh=480;
offset = 0;
if nargin<3
    resize_pos = 24;
end

if ~exist(output_posdir,'dir')
    mkdir(output_posdir);
end

finp = fopen(labelname,'r');
lli=1;
imnames = {};
rects = [];
while true
    tline = fgetl(finp);
    if ~ischar(tline)
        break;
    end
    pos = strfind(tline,'.png');
    
    imnames{lli} = tline(1:pos+3);
    rects(lli,:) = str2num(tline(pos+4:end));
    lli=lli+1;
end
rects = rects+1;

lastimname='';
ipos = 1;
for i=1:length(imnames)
    imname = imnames{i};
    rect = rects(i,:);
    if ~strcmp(lastimname,imname)
        fin = fopen(imname,'rb');
        y_gray = imread(imname);
        lastimname = imname;
        
    end
    
    xx = rect(1):rect(3);
    yy = rect(2):rect(4);
    ratio = length(xx)/length(yy);
    if (ratio>1.3 || ratio<(1/1.3))
        continue;
    end
    
    % write positive img
    posimg = get_pos_img(y_gray,rect,maxw,maxh,offset,resize_pos);
    write_pos_name = [output_posdir,'/',num2str(ipos,'%.4d'),'.png'];
    imwrite(posimg,write_pos_name);
    
    % write rotation img
    posimg_cells = get_rotated_images(y_gray,rect,maxw,maxh,offset,resize_pos);
    for j=1:length(posimg_cells)
        posimg = posimg_cells{j};
        if isempty(posimg)
            continue;
        end
        write_pos_name = [output_posdir,'/',num2str(ipos,'%.4d'),...
            '_rotate_',num2str(j,'%.3d'),'.png'];
        imwrite(posimg,write_pos_name);
    end
    ipos = ipos+1;
end



fclose(finp);
end


function [pos] = get_pos_img(y_gray,rect,maxw,maxh,offset,resize_pos)
xx = rect(1):rect(3);
yy = rect(2):rect(4);
xcutoffset = offset/resize_pos*length(xx);
ycutoffset = offset/resize_pos*length(yy);

xx = rect(1)-xcutoffset:rect(3)+xcutoffset;
yy = rect(2)-ycutoffset:rect(4)+ycutoffset;
xx = xx(xx>0);
yy = yy(yy>0);
xx = xx(xx<=maxw);
yy = yy(yy<=maxh);

pos = y_gray(yy,xx,:);
pos = imresize(pos,[resize_pos+2*offset,resize_pos+2*offset]);

end


function [cut_img_cells] = get_rotated_images( img,rect,maxw,maxh,offset,resize_pos )
cut_img_cells = {};
angle = -60:10:60;

rect_w = max([rect(3)-rect(1),rect(4)-rect(2)]);
img_center = round([maxw/2+0.5,maxh/2+0.5]);
rect_center = double(round([(rect(1)+rect(3))/2,(rect(2)+rect(4))/2]));
dirr = rect_center-img_center;

for i=1:length(angle)
    aa = angle(i);
    ty = imrotate(img,aa,'bilinear','crop');
    
    raa = aa/180*pi;
    %     rotate_matrix = [cos(aa),-sin(aa);sin(aa),cos(aa)];
    rotate_matrix = [cos(-raa),-sin(-raa);sin(-raa),cos(-raa)];
    
    after_rotate_dirr = rotate_matrix*dirr';
    after_rotate_dirr = after_rotate_dirr'+img_center;
    after_rotate_dirr = int32(round(after_rotate_dirr));
    new_rect = [after_rotate_dirr(1)-rect_w/2,after_rotate_dirr(2)-rect_w/2,...
        after_rotate_dirr(1)+rect_w/2,after_rotate_dirr(2)+rect_w/2];
    
    if new_rect(3)>maxh || new_rect(4)>maxh || new_rect(1)<1 || new_rect(2)<1
        continue;
    end
    
    cut_img_cells{i} = get_pos_img(ty,new_rect,maxw,maxh,offset,resize_pos);
    %     figure(1);
    %     imshow(ty);
    %     figure(2);
    %     imshow(cut_img_cells{i});
end

end

