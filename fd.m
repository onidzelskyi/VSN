clear all
clc
%Detect objects using Viola-Jones Algorithm

%To detect Face
FDetect = vision.CascadeObjectDetector;

%Read the input image
I = imread('photo.jpg');

%Returns Bounding Box values based on number of objects
BB = step(FDetect,I);

%figure,
%imshow(I); hold on
A = [];
for i = 1:size(BB,1)
    width = int8(BB(i,4));
    height = int8(BB(i,3));
    collum = int8(BB(i,1));
    row = int8(BB(i,2));
    A = [I(row:row+height,collum:collum+width)];
    figure,
    imshow(A); hold on; hold off;
%rectangle('Position',BB(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
end
%title('Face Detection');
%hold off;
