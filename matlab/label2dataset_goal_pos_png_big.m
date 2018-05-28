function []=label2dataset_goal_pos_png_big(labelname,output_posdir, resize_pos, add_rotate)
maxw=640;
maxh=480;
offset = 0;

if nargin<4
    add_rotate = false
end
    

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
        y_gray = imread(imname);
        lastimname = imname;
        
        yneg = y_gray;
    else
        yneg = yneg;
    end
    
    
    %% write negative img
    %     trect = double(rect);
    %     mid = [(trect(1)+trect(3))/2, (trect(2)+trect(4))/2];
    %     bottom_length = sqrt((trect(3)-trect(1))^2+(trect(4)-trect(2))^2);
    %     txx = round(mid(1)-bottom_length*2):round(mid(1)+bottom_length*2);
    %     tyy = round(mid(2)-bottom_length*2):round(mid(2)+bottom_length*2);
    %     txx = txx(txx>0);
    %     tyy = tyy(tyy>0);
    %     txx = txx(txx<=maxw);
    %     try
    %         tyy = tyy(tyy<=maxh);
    %     catch
    %         fprintf('hehe\n');
    %     end
    %     yneg(tyy,txx,:)=round(255*rand(length(tyy),length(txx),3));
    %     imshow(yneg);
    %     [~,fn,~]=fileparts(imname);
    %     write_neg_name = [output_negdir,'/',fn,'.png'];
    %     imwrite(yneg,write_neg_name);
    
    %% write positive img
    posimg = get_pos_img(y_gray,rect,maxw,maxh,offset,resize_pos);
    write_pos_name = [output_posdir,'/',num2str(ipos,'%.4d'),'.png'];
    imwrite(posimg,write_pos_name);
    
    % write rotation img
    if add_rotate
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
    end
    
    ipos = ipos+1;
end



fclose(finp);
end


function [pos] = get_pos_img(y_gray,rect,maxw,maxh,offset,resize_pos_x)
rect = double(rect);
mid = [(rect(1)+rect(3))/2, (rect(2)+rect(4))/2];
bottom_length = sqrt((rect(3)-rect(1))^2+(rect(4)-rect(2))^2);

xf = 1;
yhf = 1.5;
ylf = 0.5;
txx = round(mid(1)-bottom_length*xf):round(mid(1)+bottom_length*xf);
tyy = round(mid(2)-bottom_length*yhf):round(mid(2)+bottom_length*ylf);
xcutoffset = offset/resize_pos_x*length(txx);

resize_pos_y = (yhf+ylf)/xf/2*resize_pos_x;
ycutoffset = offset/resize_pos_y*length(tyy);

xx = round(txx(1)-xcutoffset):round(txx(end)+xcutoffset);
yy = round(tyy(1)-ycutoffset):round(tyy(end)+ycutoffset);



xx = xx(xx>0);
yy = yy(yy>0);
xx = xx(xx<=maxw);
yy = yy(yy<=maxh);

pos = y_gray(yy,xx,:);
pos = imresize(pos,[resize_pos_y+2*offset,resize_pos_x+2*offset]);

end


function [cut_img_cells] = get_rotated_images( img,rect,maxw,maxh,offset,resize_pos )
cut_img_cells = {};
angle = -6:4:6;

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