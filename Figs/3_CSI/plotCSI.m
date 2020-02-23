% Abs matters

% =========== load data =======
fileName = '~/Codes/1_CTW2/label/file_1.hdf5'

h5disp(fileName)

H_Im =  h5read(fileName,'/H_Im');
size(H_Im)
% 5 X 924 X 56 * 512(Ndat)
H_Re =  h5read(fileName,'/H_Re');



% same location
%     same channel, ant 1-5
%     same ant, channel 1-5

H_Im1 = squeeze( H_Im(:,:,:,1));
H_Re1 = squeeze( H_Re(:,:,:,1) );

H1 = ( H_Im1.^2 + H_Re1.^2 ).^0.5;

% =========== plot CSI diff Ant ========================
figure;
for i = 1:5
plot(H1(1,:,i),'o')
hold on
end
% 5 X 924 X 56 * 512(Ndat)
legend('1','2','3','4','5')
title('same location, same channel, differet Anttena ')


% ant 2 fail in previous case, additional check -> not comply
figure;
for i = 1:5
plot(H1(5,:,i),'o')
hold on
end
legend('1','2','3','4','5')
title('same location, same channel, differet Anttena ')





% =========== plot CSI diff Ant ========================

figure;
for i = 1:5
plot(H1(i,:,1),'o')
hold on
end
legend('1','2','3','4','5')
title('same location, same channel, differet channel ')



% % random pick anther one Ant 55 -> PASS
% figure;
% for i = 1:5
% plot(H1(i,:,55),'o')
% hold on
% end
% legend('1','2','3','4','5')
% title('same location, same channel, differet channel ')

