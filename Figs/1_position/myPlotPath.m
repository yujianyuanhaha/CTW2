h5disp('./file_1.hdf5')
x = h5read('file_1.hdf5','/Pos');

figure;
subplot(2,2,1)
plot(x(1,1:10), x(2,1:10),'-o')

subplot(2,2,2)
plot(x(1,1:10), x(2,1:10),'-o')
hold on;
plot(x(1,11:20), x(2,11:20),'-o')


subplot(2,2,3)
plot(x(1,1:10), x(2,1:10),'-o')
hold on;
plot(x(1,11:20), x(2,11:20),'-o')
hold on;
plot(x(1,21:30), x(2,21:30),'-o')

% ====================================
figure;
subplot(1,2,1)
scatter(x(1,:), x(2,:))

subplot(1,2,2)
scatter(x(1,:), x(2,:))
hold on;
plot(x(1,1:10), x(2,1:10),'-o')
title('first 10 position')






