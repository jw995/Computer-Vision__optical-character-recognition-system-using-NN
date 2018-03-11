function [lines, bw] = findLetters(im)
% [lines, BW] = findLetters(im) processes the input RGB image and returns a cell
% array 'lines' of located characters in the image, as well as a binary
% representation of the input image. The cell array 'lines' should contain one
% matrix entry for each line of text that appears in the image. Each matrix entry
% should have size Lx4, where L represents the number of letters in that line.
% Each row of the matrix should contain 4 numbers [x1, y1, x2, y2] representing
% the top-left and bottom-right position of each box. The boxes in one line should
% be sorted by x1 value.

% convert the image to grayscale image
I = double(rgb2gray(im));
I=I/max(I(:));

%-------------------------------------------
% best para for 01_list: char_low=400; offset=8; threshold=0.66; gap=80;
% best para for 02_letters: char_low=400; offset=25; threshold=0.6; gap=150;
% best para for 03_haiku: char_low=1000; offset=5; threshold=0.6; gap=120;
% best para for 04_deep: char_low=2000; offset=25; threshold=0.6; gap=200;
%-------------------------------------------
% set parameter for rectangular
char_low=2000; offset=25; threshold=0.6; gap=200;

I(find(I>threshold))=1;
I(find(I<=threshold))=0;
I=1-I;

% find connected elements
connect=bwconncomp(I);

figure;
imshow(im);
hold on;
idx=0;
lines={};
letters={};

% draw rectangular
for i=1:length(connect.PixelIdxList)
    element=connect.PixelIdxList{i};

    if (length(element)>char_low)
        idx=idx+1;
        x=floor(element/size(I,1))+1;
        y=mod(element,size(I,1));
        min_x=min(x);
        max_x=max(x);
        min_y=min(y);
        max_y=max(y);

        h=max_y-min_y;
        w=max_x-min_x;
        x1=min_x-offset;
        y1=min_y-offset;
        x2=x1+w+2*offset;
        y2=y1+h+2*offset;
             
        rectangle('Position',[x1,y1,w+2*offset,h+2*offset],...
                'EdgeColor','r','LineWidth',1);
        hold on;
        
        entry(idx,:)=[x1 y1 x2 y2];
        center(idx,:)=[0.5*(x1+x2) 0.5*(y1+y2)];
        center1=center;
        entry1=entry;
    end
end

index=(1:idx)';
acc=0;
group={};

% pick out the lines
while isempty(index)~=1
    acc=acc+1;
    num=0;
    sa_idx=[];
    cur_idx=[];
    for i=1:length(index)
        dis=abs(center(i,2)-center(1,2));
        if (dis<gap)
            num=num+1;           
            sa_idx(num,1)=i;
            cur_idx(num,1)=index(i,1);
        end
    end
    
    group{acc}=cur_idx;
    index(sa_idx)=[];
    center(sa_idx,:)=[];
    entry(sa_idx,:)=[];
end

% organize the line order
for i=1:length(group)
    ele=group{i};
    avg_y(1,i)=mean(center1(ele,2));
end

[~,rank]=sort(avg_y);

for i=1:length(group)
    regroup{i}=group{rank(1,i)};
    temp=[];
    for j=1:length(regroup{i})
        p=regroup{i};
        temp(j,:)=entry1(p(j,1),:);
        letters{i}=temp;
   end
end


% get the characters in a sequence
idx=0;
for i=1:length(letters)
    letter=letters{i};
    for j=1:size(letter,1)
        idx=idx+1;
        character=letter(j,:);
        x1=character(1,1);
        y1=character(1,2);
        x2=character(1,3);
        y2=character(1,4);
        
        wid=abs(x2-x1);
        hei=abs(y2-y1);
        dif=wid-hei;
        
         if (dif>0)
            y1=y1-0.5*dif;
            y2=y2+0.5*dif;    
        else 
            x1=x1+0.5*dif;
            x2=x2-0.5*dif;           
         end  

         if (y1<0)
             y1=1;
         end
         lines{idx}=1-I(floor(y1):floor(y2),floor(x1):floor(x2)); 

    end
end

bw=I;

end
