# LMM-CMA-ES reproduction

## 1. Introduction
This experiment serves as an introduction to further research in my master's thesis. Here the results of LMM-CMA-ES algorithm are reproduced to ensure that the metamodel is implemented correctly as it will be used in further research. LMM-CMA-ES uses approximate ranking metamodel which uses a polynomial regression approximator.

In this experiment the metamodel will be reproduced according to the publication, and the CMA-ES implementation used will be the official python one made by Nikolas Hansen, the author of the publication.

The experiment will be considered a success if the reproduced results will be similar to those reported in the paper.

## 2. Problem Definition and Algorithm

### 2.1 Task Definition

Precisely define the problem you are addressing (i.e. formally specify the inputs and outputs). Elaborate on why this is an interesting and important problem.

### 2.2 Algorithm Definition

Describe in reasonable detail the algorithm you are using to address this problem. A psuedocode description of the algorithm you are using is frequently useful. Trace through a concrete example, showing how your algorithm processes this example. The example should be complex enough to illustrate all of the important aspects of the problem but simple enough to be easily understood. If possible, an intuitively meaningful example is better than one with meaningless symbols.

## 3. Experimental Evaluation

### 3.1 Methodology
Functions used for performance comparisons are Cumulative Squared Sums function (incorectly called Schwefel function in the publication) and Rosenbrock function. They are respectively unimodal and multimodal, which makes the pair a good choice to measure the algorithm in different function modalities.


The metric used for comparison and benchmarking is the number of objective function evaluations. The average and standard deviation of 20 runs will be used.


ecdf curve
hyperparams

### 3.2 Results
The experiment results are expressed in a form of a table. The mean and standard deviation values are rounded to the closest whole number, and separated with a plus-minus sign to minize the number of columns in the table.

#### Cumulative Squared Sums function
Bounds: [-10, 10]^n

| n | popsize | LMM reproduced | LMM reported | CMA-ES reproduced | CMA-ES reported |
|---|---------|----------------|--------------|-------------------|-----------------|
| 2 | 6       | 62 ± 4         | 81 ± 5       | 407 ± 36          | 391 ± 42        |
| 4 | 8       | 123 ± 10       | 145 ± 7      | 926 ± 45          | 861 ± 53        |
| 8 | 10      | 232 ± 11       | 282 ± 11     | 2070 ± 100        | 2035 ± 93       |

#### Rastrigin function
Bounds: [-5, 5]^n

| n | popsize | LMM reproduced | LMM reported | CMA-ES reproduced | CMA-ES reported |
|---|---------|----------------|--------------|-------------------|-----------------|
| 2 | 6       | 205 ± 53       | 263 ± 87     | 764 ± 150         | 799 ± 119       |
| 4 | 8       |                | 674 ± 103    |                   | 1973 ± 291      |
| 8 | 10      |                | 2494 ± 511   |                   | 6329 ± 747      |

### 3.3 Discussion

Is your hypothesis supported? What conclusions do the results support about the strengths and weaknesses of your method compared to other methods? How can the results be explained in terms of the underlying properties of the algorithm and/or the data.

## 4. Related Work

Answer the following questions for each piece of related work that addresses the same or a similar problem. What is their problem and method? How is your problem and method different? Why is your problem and method better?

## 5. Future Work
As the approximate ranking metamodel was successfuly reproduced, the next step would be using K nearest neighbours approximator instead of polynomial regression, and reproducing Konrad Krawczyk's results from similar experiments on JADE algorithm.

## 6. Conclusion
Results suggest that the metamodel was successfuly reproduced. Experiment results are slightly better than those presented in the publication, which might be due to

## Bilbiography
- [LMM-CMA-ES publication](http://www.cmap.polytechnique.fr/~nikolaus.hansen/ppsn06model.pdf)