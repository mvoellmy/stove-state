clear all

path = 'data/In-airGestures/Training/gesture1/CleanSegmentation';


data = [];
labels = [];

for i=101:150
    data(:,:,i) = imread(fullfile(path , strcat('tip', int2str(i), '.png')));
    labels(:,:,i) = imread(fullfile(path , strcat('tip', int2str(i), '.png')));
end

save('myData.mat', 'data', 'labels');