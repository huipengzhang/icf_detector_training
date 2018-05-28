function []=label_target_goal_png(inputdirname,outputfile)
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
    
    xx={};
    yy={};
    temp = y_gray;
    label_ind = 0;
    while true       
% way 1, using my naive function to crop images
        [x,y,b] = ginput(2);
        if(length(x)~=2)
            break;
        else   
            if label_ind~=0
                cinput = input('Delete Previous One? (y/n) \n','s');
                if isempty(cinput)||(cinput=='n')
                    label_ind = label_ind+1;
                end
            else
                label_ind = label_ind+1;
            end
            
            xx{label_ind}=int32(round(x));
            yy{label_ind}=int32(round(y));
            
%             if(xx(1)>=xx(2) || yy(1)>=yy(2))
%                 continue;
%             end
            


            temp = y_gray;
            
            for li=1:length(xx)
                
                if xx{li}(1)<=xx{li}(2)
                    w = xx{li}(1):xx{li}(2);
                else
                    w = xx{li}(2):xx{li}(1);
                end
                
                if yy{li}(1)<=yy{li}(2)
                    h = yy{li}(1):yy{li}(2);
                else
                    h = yy{li}(2):yy{li}(1);
                end                
                
                
                temp(h,w,1)=255;
                temp(h,w,2)=0;
                temp(h,w,3)=0;
            
            end
            
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
    
    for li=1:length(xx)
        txx = xx{li};
        tyy = yy{li};
        if(txx(1)>0 && tyy(1)>0)
            txx = txx-1;
            tyy = tyy-1;
            fprintf(fout,'%s %d %d %d %d\n',imname,txx(1),tyy(1),txx(2),tyy(2));% 0-based coordinate
        end
    end
end

fclose(fout);


end