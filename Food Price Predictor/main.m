clc;
clear all;
close all;
warning off;
[file,path]=uigetfile('*.*');
a1=imread([path,file]);
figure;
imshow(a1);
title('input image');

matlabroot='D:\CODE';
datasetpath = fullfile(matlabroot,'Traindata');
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');

net=alexnet;
layers = net.Layers;
layers(23) = fullyConnectedLayer(3,'Name','fc');
layers(25) = classificationLayer('Name','CcL'); 

imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, imds); 

options = trainingOptions('sgdm', 'MaxEpochs', 20,'initialLearnRate',0.0001);    
convnet = trainNetwork(augmentedTrainingSet,layers,options);

% load net
bw=im2bw(a1);
[Ilabel, num]= bwlabel(bw,8);
Iprops = regionprops(bw,'BoundingBox','Area');
Ibox=[Iprops.BoundingBox];
Ibox=reshape(Ibox,[4 num]);

set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
hold on;
for k=1:num
   if Iprops(k).Area> 3000      
      thisBlobsBoundingBox = Iprops(k).BoundingBox; 
      subImage = imcrop(a1, thisBlobsBoundingBox); 
      a2=imresize(subImage,[227 227]);
      YPred{1,k} = classify(convnet,a2);
      rectangle('position',Ibox(:,k),'edgecolor','g');
      text(thisBlobsBoundingBox(1)+thisBlobsBoundingBox(3)/2,thisBlobsBoundingBox(2)+thisBlobsBoundingBox(4)/2,char(YPred{1,k}),'Color','green','Fontsize',15);
   end
end
hold off;
j=1;
for k=1:numel(YPred)
    if isempty(YPred{1,k})==0
        x=string(YPred{1,k});
        if x=='samosa'
            cost1(j)=20;
        elseif x=='friedrice'
            cost1(j)=100;
        elseif x=='pizza'
            cost1(j)=199;
        else
        end
        j=j+1;
    end
end
Totalcost=sum(cost1);
disp('Totalcost=');
disp(Totalcost);