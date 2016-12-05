
%• How many clusters there are in the end. (A cluster can “disappear” in one iteration of the
%algorithm if no vectors or closest to its centroid.)
%• The final centroids of each cluster.
%• The number of pixels associated to each cluster.
%• The sum of squared distances from each pixel to the nearest centroid after every iteration of the
%algorithm.

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

data = load("hw4-image.txt");

m = length(data(:,1));
d = length(data(1,:));
k = length(init_matx(:,1));

% guesstimated number of iterations for preallocation
ITERS = 50;

k_means = repmat(reshape(init_matx', 1, d, k), m, 1, 1);
new_means = init_matx;
k_labels = 1:m;
k_settled = false;
k_membership_counts = zeros(ITERS, k);
means_trajectory = zeros(k,d,ITERS);
sum_squared_dist = zeros(1, ITERS);
iteration_count = 1;
% keep k-copies of the data to quickly get the distance to centroids
unfolded_data = repmat(data, 1, 1, k);
% get the L2 norm of the row-slice of the difference between 8 pixel copies and centroids
slice_norm = @(tensor)reshape(sum((tensor.^2),2).^0.5, size(tensor, 1), size(tensor, 3));

% Repeat the process until done...
while (~k_settled)
  loop = tic();
  disp(sprintf('Clustering all pixels, iteration %d', iteration_count));
  fflush(stdout);
  k_means = repmat(reshape(new_means', 1, d, k), m, 1, 1);
  k_means_flat = reshape(k_means(1,:,:), d, k)';
  [min_l2, idxs] = min(slice_norm(unfolded_data - k_means), [], 2);
  sum_squared_dist(iteration_count) = sum(min_l2.^2);
  k_labels = idxs;
  disp(sprintf('Reassigning means'));
  fflush(stdout);
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
  fflush(stdout);  
  disp(sprintf('iteration %d took %d seconds.', iteration_count, toc(loop)));
  fflush(stdout);
  
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

% Give some info
disp(sprintf('There are %d total clusters with pixels in them',...
             sum(k_membership_counts(iteration_count,:) ~= 0) ));

disp('Cluster membership count is respectively:');
disp(k_membership_counts(iteration_count,:));

disp('The final centroids are:');
disp(k_means_flat)

plot(sum_squared_dist)
title('sum of squared distances to centroid per iteration');
xlabel('iterations');

disp(sprintf('The whole process took %d seconds.', toc(t0)));
