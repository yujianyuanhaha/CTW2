% fingerprint system check by DNN on small dataset
% `all channel are the same`, 1/5 data is enough

Xraw =[];
Y = [];

tic;
for i = 1:9
    
    
    fileName = strcat('~/Codes/1_CTW2/label/file_',...
    num2str(i),'.hdf5')

H_Im =  h5read(fileName,'/H_Im');
H_Re =  h5read(fileName,'/H_Re');

H = ( H_Im.^2 + H_Re.^2 ).^0.5;
H1 = squeeze(H(1,:,:,:));  % 1/5 is enough

y0 = h5read(fileName,'/Pos');
y = y0(1:2,:);


Xraw = cat(3,Xraw,H1);
Y = [Y,y];

toc;

end

XfileName = dir( 'X.mat');
XfileName.bytes/(1024^2)
% 42M data; while x8 for unlabelled, not heavy tho

figure;
scatter(Y(1,:),Y(2,:));

% resort
X = permute(X,[3,2,1]);
save('X.mat','X');
Y = Y';
save('Y.mat','Y');

% split
Xtest = X(1:1000,:,:);
Ytest = Y(1:1000,:,:);
Xtrain = X(1001:end,:,:);
Ytrain = Y(1001:end,:,:);
save('Xtrain.mat','Xtrain')
save('Ytrain.mat','Ytrain')
save('Xtest.mat','Xtest')
save('Ytest.mat','Ytest')


% plot
figure;
subplot(1,2,1);
scatter(Ytrain(:,1),Ytrain(:,2))
title('trained 3979 positions')
subplot(1,2,2);
scatter(Ytest(:,1),Ytest(:,2))
title('test 1000 positions')


