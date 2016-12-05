% COMP 6321 Machine Learning, Fall 2016
% Federico O'Reilly Regueiro - 40012304
% Assignment 4, due December 9th


% Script file for performing k-means clustering. Should return the following:
%• How many clusters there are in the end. (A cluster can “disappear” in one iteration of the
%algorithm if no vectors or closest to its centroid.)
%• The final centroids of each cluster.
%• The number of pixels associated to each cluster.
%• The sum of squared distances from each pixel to the nearest centroid after every iteration of the
%algorithm.

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
  flush = @()fflush(stdout);
else
  flush = @()drawnow('update');
end

t0 = tic();

init_matx = ...
[255, 255, 255; ...
255, 0, 0; ...
128, 0, 0; ...
0, 255, 0; ...
0, 128, 0; ...
0, 0, 255; ...
0, 0, 128; ...
0, 0, 0];

data = load('hw4-image.txt');

m = length(data(:,1));
d = length(data(1,:));
k = length(init_matx(:,1));

% guesstimated number of iterations for preallocation, 
% This is mostly for efficiency as the loop will stop before if converged
ITERS = 50;

% Initial centroids or cluster means
k_means = repmat(reshape(init_matx', 1, d, k), m, 1, 1);
new_means = init_matx;
k_labels = 1:m;
k_settled = false;
% how many pixels per cluster per iteration
k_membership_counts = zeros(ITERS, k);
% keep a trace of the centroids per iteration
means_trajectory = zeros(k,d,ITERS);
sum_squared_dist = zeros(1, ITERS);
iteration_count = 1;
% keep k-copies of the data to quickly get the distance to centroids
unfolded_data = repmat(data, 1, 1, k);
% get the L2 norm of the row-slice of the difference between 8 pixel copies and centroids
slice_sq_norm = @(tensor)reshape(sum((tensor.^2),2), size(tensor, 1), size(tensor, 3));

% Repeat the process until done... 
while (~k_settled)
  loop = tic();
  disp(sprintf('Clustering all pixels, iteration %d', iteration_count));
  flush();
  k_means = repmat(reshape(new_means', 1, d, k), m, 1, 1);
  k_means_flat = reshape(k_means(1,:,:), d, k)';
  [min_l2_sq, idxs] = min(slice_sq_norm(unfolded_data - k_means), [], 2);
  sum_squared_dist(iteration_count) = sum(min_l2_sq);
  k_labels = idxs;
  disp(sprintf('Reassigning means'));
  flush();
  for kth = 1:k
      temp = zeros(size(data));
      kth_pixels_idx = (k_labels == kth);
      temp(kth_pixels_idx, :) = data(kth_pixels_idx, :);
      temp(~kth_pixels_idx, :) = [];
      k_membership_counts(iteration_count, kth) = size(temp, 1);
      if (size(temp, 1) >= 1)
        new_means(kth, :) = mean(temp);
      end
  end
  means_trajectory(:,:,iteration_count) = k_means_flat;
  disp(sprintf(...
    'Norm of the difference between this iteration and last''s means: %d',...
     norm(new_means-k_means_flat)))
  flush();
  disp(sprintf('iteration %d took %d seconds.', iteration_count, toc(loop)));
  flush();
  
  % See if we're done
  if (sum(sum(new_means ~= k_means_flat)) == 0)
    k_settled = true;
  else
    iteration_count = iteration_count + 1;
  end
  
end

% Now convert pixels to their respective centroid
clustered = data;

for kth = 1:k
      clustered(k_labels == kth, :) = k_means(k_labels == kth, :, kth);
end

% Write the file
R = reshape(clustered(:,1)./255, 407, 516);
G = reshape(clustered(:,2)./255, 407, 516);
B = reshape(clustered(:,3)./255, 407, 516);
the_image = cat(3, R', G', B');
fliplr(the_image);
imwrite(the_image, 'much_better_than_trump_patches.jpg');

% cut pre-allocated arrays if they weren't fully used
k_membership_counts(iteration_count+1:end, :) = [];
means_trajectory(:,:,iteration_count+1:end) = [];
sum_squared_dist(:, iteration_count+1:end) = [];

disp(sprintf('The whole process took %d seconds.', toc(t0)));

plot_time = tic();
% Give some info
disp(sprintf('There are %d total clusters with pixels in them',...
             sum(k_membership_counts(iteration_count,:) ~= 0) ));

disp('Final cluster membership count is respectively:');
disp(k_membership_counts(iteration_count,:));

disp('The final centroids are:');
disp(k_means_flat)

% now plot some things
g = figure(1);
for iter = iteration_count:-1:1
  for kth = 1:k
    color = means_trajectory(kth, :, iter)./255;
    h = barh(kth, iter, 'stacked', 'faceColor', color);
    hold on;
  end  
end
hold off;
axis([0,iteration_count,0,k+1]);
title('Colors resulting from centroid movement over iterations')
xlabel('Iterations');
ylabel('Cluster');
%saveas(gcf,'centroidEvolutions.pdf');

figure(2)
plot(sum_squared_dist)
title('sum of squared distances to centroid per iteration');
xlabel('iterations');
%saveas(gcf,'sumSquaredDistances.pdf');

figure(3)
for kth = 1:k
  subplot(ceil(k/2), 2, kth);
  plot(k_membership_counts(:, kth));
  hold on;
end
hold off;

figure(4)
legends = {};
hold on;
for iter = iteration_count:-1:1
  for kth = 1:k
    %subplot(ceil(k/2), 2, kth);
    color = means_trajectory(kth, :, iter)./255;
    modif = (kth-(k/2))/k;
    iter_modif = iter + modif;
    plot(iter_modif, k_membership_counts(iter, kth), 'o', 'color', color,...
                  'MarkerEdgeColor','k',...
                  'MarkerFaceColor',color);
  end
end
for kth=1:k
  legends{kth} = sprintf('Cluster %d', kth);
end
hold off;
legend(legends);
title('Cluster number of pixels and color over iterations')
xlabel('Iterations');
ylabel('Number of pixels');
%saveas(gcf,'clusterMembership.pdf');

disp(sprintf('Plotting took %d seconds', toc(plot_time)));
