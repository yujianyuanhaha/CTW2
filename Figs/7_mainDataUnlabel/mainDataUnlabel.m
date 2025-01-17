% 7 section
% 1. load    2. make X Y   3. smooth X   4. classify Y

% ==============  1. load ===============

ID = 6

Xraw =[];
Y = [];

tic;
for i = 0:4
    
    
    fileName = strcat('~/Codes/1_CTW2/unlabel/file_',...
    num2str(ID), num2str(i),'.hdf5')

H_Im =  h5read(fileName,'/H_Im');
H_Re =  h5read(fileName,'/H_Re');

H = ( H_Im.^2 + H_Re.^2 ).^0.5;
H1 = squeeze(H(1,:,:,:));  % 1/5 is enough

% y0 = h5read(fileName,'/Pos');
% y = y0(1:2,:);


Xraw = cat(3,Xraw,H1);
% Y = [Y,y];

toc;

end

% XfileName = dir( 'X.mat');
% XfileName.bytes/(1024^2)
%42M data; while x8 for unlabelled, not heavy tho



Xraw = permute(Xraw,[3,2,1]);
save(strcat('Xraw',num2str(ID),'.mat'),'Xraw'); % ==============

% ==============  3 smooth X ===============
Ndat = size(Xraw,1);

warning off

tic
for i = 1:56
    for j = 1:Ndat
        
        t = squeeze( Xraw(j,i,:) )';
        p = polyfit(1:924,t,10);
        t_h = polyval(p,1:924);
        
        X(j,i,:) = t_h(1:14:end);
    end
    disp(i)
    toc;
end
 

save(strcat('X',num2str(ID),'.mat'),'X')



