load('Ytest.mat')
load('/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/512x3_p8/Ypred_TF.mat')

mean( abs(Ypred- Ytest))
% 45.2426   73.6470

% x < 200, y < -400
idx= [];
for i = 1:1000
    if Ytest(i,1) > 200 && Ytest(i,2) > -300
        idx = [idx,i];
    end
end

mean( abs(Ypred(idx,:)- Ytest(idx,:)  ))
% 50.4150   66.7411