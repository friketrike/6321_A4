
%• How many clusters there are in the end. (A cluster can “disappear” in one iteration of the
%algorithm if no vectors or closest to its centroid.)
%• The final centroids of each cluster.
%• The number of pixels associated to each cluster.
%• The sum of squared distances from each pixel to the nearest centroid after every iteration of the
%algorithm.

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

k_means = init_matx;
k_labels = 1:length(data(:,1));
k_settled = false;
k_membership_counts = zeros([], length(init_matx(:,1)));
means_trajectory = zeros(8,3,[]);
iterations = 1;
while (~k_settled) 
  tic();
  disp(sprintf('Clustering all pixels, iteration %d', iterations));
  fflush(stdout);
  for l = 1:length(data(:,1))
    temp_pix = repmat(data(l,:), 8, 1);
    [min_l2, idx] = min(norm(temp_pix - k_means, 2, "rows"));
    k_labels(l) = idx;
  end
  new_means = k_means;
  disp(sprintf('Reassigning means'));
  fflush(stdout);
  for n = 1:length(k_means(:,1))
      idx = (k_labels == n);
      k_membership_counts(iterations, n) = sum(idx);
      if (sum(idx) > 0)
        new_means(n,:) = mean(data(idx,:));
      end
  end
  if (new_means == k_means)
    k_settled = true;
    clustered = data;
    for n = 1:length(k_means(:,1))
      idx = (k_labels == n);
      if sum(idx > 0)
        clustered(idx, :) = repmat(k_means(n, :), sum(idx), 1);
      end
    end
  else
    means_trajectory = cat(3, means_trajectory, k_means);
    disp(sprintf('Norm of the difference between this iteration and last''s means: %d', norm(new_means-k_means)))
    fflush(stdout);
    k_means = new_means;
  end
  disp(sprintf('iteration %d took %d seconds.', iterations, toc()));
  fflush(stdout);
  iterations = iterations + 1;
end
R = reshape(clustered(:,1)./255, 407, 516);
G = reshape(clustered(:,2)./255, 407, 516);
B = reshape(clustered(:,3)./255, 407, 516);
the_image = cat(3, R', G', B');
fliplr(the_image);
imwrite(the_image, 'much_better_than_trump_patches.jpg');
