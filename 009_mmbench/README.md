# mmbench
This project aims to benchmark against each other all the surrogate models we have in optilab package. Specifically we want to benchmark their interpolation and extrapolation abilities in metamodeling for CMA-ES variants. To achieve this, we first create a representative dataset with states sampled from real ipop cma es algorithm on cec 2013 functions. Each state consists of centerpoint M, standard deviation SIGMA and covariance matrix C. To sample these states we use sampler_ipop_cma_es algorithm.

We sample 51 states per function per dimensionality. There are 28 functions 1-28, and dimensionalities 10 and 30. we should store this data in a JSON for each dimensionality.

Then we have two scripts for testing interpolation and extrapolation abilities. For each state we shall generate 2*N points from the distribution, like in CMA-ES population generation. Then in interpolation we split this set into two N-size sets: train and test. for extrapolation, we first sort these points by mahalanobis distance, and take the closer half is the train set, and the further half is test set.

What is the expected value of the boundary dividing the extrapolation sets expressed as the mahalanobis distance?

What is the best N to use here? make it a default.

Then for each each metamodel we run either interpolation or extrapolation on all 51 states. then we collect both median and mean MAPE and spearman rank corellation. We store it in CSV file. separately for interpolation and extrapolation.

Finally we calculate for each metamodel the average medians of MAPE and spearman on all28 functions for a given dimensionality.