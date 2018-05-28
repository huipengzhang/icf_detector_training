function []=label_target_ball_png(inputdirname,outputfile)
if nargin<2
    outputfile = 'label.txt';
end
if nargin<1
    inputdirname = './';
end


fout = fopen(outputfile,'w');

content = dir([inputdirname,'/*.png']);
for i=1:length(content)
    imname = [inputdirname,'/',content(i).name];
    y_gray = imread(imname);

    imshow(y_gray);    
    
    xx=[-1,-1];
    yy=[-1,-1];
    temp = y_gray;
    while true       
% way 1, using my naive function to crop images
        [x,y,b] = ginput(2);
        if(length(x)~=2)
            break;
        else
            xx=int32(round(x));
            yy=int32(round(y));
            if(xx(1)>=xx(2) || yy(1)>=yy(2))
                continue;
            end
            
            temp = y_gray;
            
            w = xx(1):xx(2);
            h = yy(1):yy(2);            
            
            temp(int32(repmat(yy(1),length(w),1)),w,1)=255;
            temp(int32(repmat(yy(2),length(w),1)),w,1)=255;
            temp(h,int32(repmat(xx(1),length(h),1)),1)=255;
            temp(h,int32(repmat(xx(2),length(h),1)),1)=255;
            imshow(temp);
        end

        % way 2, using matlab's function to crop images
%         [~,r] = imcrop(temp);
%         if isempty(r)
%             break;
%         else
%             temp = y_gray;
%             r = int32(round(r));
%             
%             xx = [r(1),r(3)+r(1)];
%             yy = [r(2),r(2)+r(4)];
%             
%             w = xx(1):xx(2);
%             h = yy(1):yy(2);     
%             
%             temp(int32(repmat(yy(1),length(w),1)),w,1)=255;
%             temp(int32(repmat(yy(2),length(w),1)),w,1)=255;
%             temp(h,int32(repmat(xx(1),length(h),1)),1)=255;
%             temp(h,int32(repmat(xx(2),length(h),1)),1)=255;
%             imshow(temp);
%         end
    end
    
    if(xx(1)>0 && yy(1)>0)
        xx = xx-1;
        yy = yy-1;
        fprintf(fout,'%s %d %d %d %d\n',imname,xx(1),yy(1),xx(2),yy(2));% 0-based coordinate
    end
end

fclose(fout);


end