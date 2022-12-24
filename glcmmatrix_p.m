%zahra farzadpour
%using GLCM Matrix to detect real and fake fingerprint
close all
clear all
clc
%read input images(for train)
imagepath5='train';
filelist5=dir(fullfile(imagepath5,'*.bmp'));
list5={filelist5.name};
for i=1:length(list5)
    img5{i,1}=imresize(imread(fullfile(imagepath5,list5{i})),[96 96]); 
    
end
data_train=[img5];
for i=1:120
    input1=data_train{i,1};
     %glcm matrix
glcm = graycomatrix(input1);
glcm_input{i}=glcm;
%calcute feature from glcm matrix
stats = graycoprops(glcm);
stats_input{i}=stats;
f=stats_input{i};
v=f(1,1);
e(i,:)=v;
end
p=struct2cell(e);
pp=cell2mat(p);
xdata=[pp]';
%label
for q=1:60
    group{q,1}='real';
end
for q=61:120
    group{q,1}='fake';
end
%svm struct
svmStruct= svmtrain(xdata,group,'kernel_function','rbf','rbf_sigma',1,'showplot',false);
%testing 
%read input images
%input images for testing
imagepath1='test';
filelist1=dir(fullfile(imagepath1,'*.bmp'));
list1={filelist1.name};
for i=1:length(list1)
    img1{i,1}=imresize(imread(fullfile(imagepath1,list1{i})),[96 96]); 
    
end
data_test=[img1];
for i=1:400
    input1=data_test{i,1};
    %produce a block random matrix
    aa=randi(96);
    bb=randi(96);
    for x=aa:(aa+40-1)
        for y=bb:(bb+40-1)
            input1(x,y)=0; 
        end
    end
    MissImage1{i,1}=input1;
end
for i=1:400
    input11=MissImage1{i,1};
glcm1 = graycomatrix(input11);
glcm_input1{i}=glcm1;
%calcute feature from glcm matrix
stats1 = graycoprops(glcm1);
stats_input1{i}=stats1;
f1=stats_input1{i};
v1=f1(1,1);
e1(i,:)=v1;
end
p1=struct2cell(e1);
pp1=cell2mat(p1);
sample=[pp1]';
%svm test
Test = svmclassify(svmStruct,sample,'showplot',false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%f-score
for i=1:300
    actual{i,1}=[0];
end
for i=301:400
    actual{i,1}=[1];
end
for i=1:400
    if Test{i,1}=='real'
        predicted{i,1}=[0];
    else
        predicted{i,1}=[1];
    end
end
ACTUAL=(cell2mat(actual));
PREDICTED=(cell2mat(predicted));
result=fscore(ACTUAL,PREDICTED);
