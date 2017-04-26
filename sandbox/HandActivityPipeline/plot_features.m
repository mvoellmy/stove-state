% Plot features of hand gestures - BEGG

clear all
close all
clc

path_features = 'gesture_features/begg/';
files = dir(fullfile(path_features, '*.csv'));
num_files = 4;
num_gestures = 6;
gesture_names = {'place pan', 'place egg', 'place lid', 'remove lid', 'remove egg', 'remove pan'};
 

outs = cell(numel(files),1);
for i=1:length(files)
    outs{i} = dlmread(fullfile(path_features, files(i).name), ' ');

end

for i=1:num_gestures
    subplot(2,3,i)
    
    for j=1:num_files
        plot(outs{(i-1)*num_files+j}(:,2), -outs{(i-1)*num_files+j}(:,1), '-')
        hold on
    end
    title(gesture_names{i})
    legend('show')
end

figure()
for i=1:num_gestures
    subplot(2,3,i)
    
    for j=1:num_files
        vel = sqrt( (outs{(i-1)*num_files+j}(2:end,4)).^2 + (outs{(i-1)*num_files+j}(2:end,3)).^2 );
        plot(1:length(vel), vel, '-')
        hold on
    end
    title(gesture_names{i})
    legend('show')
end

figure()
for i=1:num_gestures
    subplot(2,3,i)
    
    for j=1:num_files
        orientation = outs{(i-1)*num_files+j}(:,5);
        plot(1:length(orientation), orientation, '-')
        hold on
    end
    title(gesture_names{i})
    legend('show')
end
