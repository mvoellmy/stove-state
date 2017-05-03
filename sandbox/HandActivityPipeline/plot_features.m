% Plot features of hand gestures

clear all
close all
clc

path_features = 'gesture_features/begg/';
files = dir(fullfile(path_features, '*.csv'));
num_videos = 7;
num_gestures = 6;
gesture_names = {'place pan', 'place egg', 'place lid', 'remove lid', 'remove egg', 'remove pan'};
% gesture_names = {'place pan', 'pour oil', 'place egg', 'stirr', 'season salt', 'remove pan'}; 

outs = cell(numel(files),1);
for i=1:length(files)
    outs{i} = dlmread(fullfile(path_features, files(i).name), ' ');
end

for i=1:num_gestures
    subplot(2,3,i)
    
    for j=1:num_videos
        plot(outs{(i-1)*num_videos+j}(:,2), -outs{(i-1)*num_videos+j}(:,1), 'x-')
        hold on
        if i==1
            plot(outs{(i-1)*num_videos+j}(1,2), -outs{(i-1)*num_videos+j}(1,1), 'o')
        end
    end

    title(gesture_names{i})
    legend('show')
end
i=1;
figure()
for j=1:num_videos
    plot(outs{(i-1)*num_videos+j}(:,2), -outs{(i-1)*num_videos+j}(:,1), 'x-')
    hold on
    if i==1
        plot(outs{(i-1)*num_videos+j}(1,2), -outs{(i-1)*num_videos+j}(1,1), 'o')
    end
end
legend('show')
% figure()
% for i=1:num_gestures
%     subplot(2,3,i)
%     
%     for j=1:num_videos
%         vel = sqrt( (outs{(i-1)*num_videos+j}(1:end,4)).^2 + (outs{(i-1)*num_videos+j}(1:end,3)).^2 );
%         plot(1:length(vel), vel, '-')
%         hold on
%     end
%     title(gesture_names{i})
%     legend('show')
% end
% 
% figure()
% for i=1:num_gestures
%     subplot(2,3,i)
%     
%     for j=1:num_videos
%         orientation = outs{(i-1)*num_videos+j}(:,5);
%         plot(1:length(orientation), orientation, '-')
%         hold on
%     end
%     title(gesture_names{i})
%     legend('show')
% end

N = 10;
spatio_temporal_features = [];
for i=1:length(outs)
    step = floor(length(outs{i}) / N);
    features = atan2(-outs{i}(1:step:step*N,3), ...
        outs{i}(1:step:step*N,4)); 
    spatio_temporal_features(end+1,:) = features(1:end);
end
figure()
C = [0,0,1;
     0,1,0;
     0,1,1;
     1,0,0;
     1,0,1;
     1,1,0];
for i=1:num_gestures
    
    for j=1:num_videos
        s(i) = scatter(1:N, spatio_temporal_features((i-1)*num_videos+j,:), ...
            50, C(i,:), 'filled','DisplayName',gesture_names{i});
        hold on
    end
end
legend(s,gesture_names)

%% Cluster features
k = 10;
clusters = kmeans(spatio_temporal_features, k)
clusters = reshape(clusters, [num_gestures, num_videos])