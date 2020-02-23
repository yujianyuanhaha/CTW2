load('/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/Ytest.mat')
load('/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/RF_Classify/Ypred_RF.mat')

figure; 
scatter(Ytest(:,1),Ypred(:,1),1)
hold on;
scatter(Ytest(:,2),Ypred(:,2),1)


figure; 
subplot(1,2,1);
scatter(Ytest(:,1),Ypred(:,1),1)
subplot(1,2,2);
hold on;
scatter(Ytest(:,2),Ypred(:,2),1)