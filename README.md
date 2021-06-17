# cancer_markers
The main aim of this repository is presenting a range of combinatorial algorithms for the identification of optimal 
combinations of biomarkers. All programs are very adaptable to prognostic, clinical, biological and any study/project 
that require the investigations of large number of variables.

## Install requirements

    pip install -r requirements.txt

## Data
This folder includes the raw data of markers used to develop these programs. Unfortunately, due to ethical constraints, 
the survival and clinical dat can not be shared or made public. The validation cohort used
to validate the marker signature is also provided https://www.ncbi.nlm.nih.gov/geo/geo2r/?acc=GSE39582.
Additional demo data file was included to
demonstrate the functionality of these algorithms. The demo data was downloaded from https://github.com/sebp/
scikit-survival/tree/master/sksurv/datasets/data.

## Survival libraries
To asses survival associations, codes in this repository used Lifelines 
(https://lifelines.readthedocs.io/en/latest/Quickstart.html), and sksurv (https://scikit-survival.readthedocs.
io/en/latest/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html).
Lifelines library is especially useful because of the range of parameters, functions and outputs available.

## Algorithm code files
### Combinatorial algo.py
This algorithm compares combinations generated through n! / r! * (n - r)! with 
r = [n, n-s, n-2s,…, 2], step (s) ∈ {1- n-1}.  Using the initial biomarker population, 
combinations are generated and evaluated to select a combination of biomarkers as the best 
solution, then the biomarkers in the selected combination are used to generate a new set of 
combinations and a new combination is selected as the new best solution. The same process is 
repeated until convergence. If s=1 then combinations will always have size n-1 which means one
biomarker will be eliminated each iteration until convergence. If the size of the initial
population is small, then larger step (s) is preferred because it has more global approach.

### combinatorial_cluster.py
The same as combinatorial except it only compute composite variable using a cluster parametr.

### mutation.algo.py/mutation_crossover.py
A number of combinations are selected through binomial random selection (1: the marker is 
included and 0: the markers is excluded), fitness is assessed, “parent” 1 along “parent”  are 
selected through roulette or tournament, a first “child” is computed
through an intersection of both “parents”, mutations/crossovers are then executed on the first 
child to generate further “children”. Then all children are 
evaluated, and the fittest child is selected for the next generation which is exposed to the 
same process until a condition is met or convergence. 

### feature selection.py
 This has been adapted from Pedregosa, F. et al. Scikit-learn: Machine learning in Python (2011).
 Firstly, compute the accuracy of a multivariate model which includes all biomarkers; then 
the clusters of correlated biomarkers are computed based on Spearman rank-order 
correlations and one biomarker is removed from each cluster.
The programme will iterate over all biomarkers in each cluster and remove 
biomarkers with the least impact on the accuracy of the model after their removal.

### Demo.py
This is a demo file of the combinatorial algo.py using the demo data file.
The combinatorial algo has to main functions implementing a gradient and global approaches.
Both functions search for best combination from a list of variables. If total model is True: the function 
will assess the impact of variables as a model rather than an individual unit. Missing values can be all
dropped or interpolated using range of methods. Composite_compute defines how the combination is assessed:
one composite variable is computed through clustering of variables or through a linear equation.

### Instructions
1-After downloading/copying any code, functions are called with default/ adapted parameters. 
For example, calling 'optimal_comb_stepwise_global(data, markers)' will implement global combinatorial
algorithm using default parameters and calling 'optimal_comb_stepwise_global(data, markers,total_model=False)' 
will implement the same algorithm, but the performance of combination alone is assessed rather than whole model.
Adjusting the parameter Sac_rate is important for not allowing the program to get stuck with a solution.
2-Make sure the data is downloaded and/or path is adjusted accordingly.
3- If you want to test on your own data, make sure to change the following elements within the code to suit your
variables name: Survival (Time) and Status (Event) and all clinical/established variables (agerecoded,Tstagen,Nstagen 
and EMVI) you want to test your own variables against. IF there are no established clinical variables, then call the 
function by specifying multi=False.
4- IF your data includes large number of variables, co-linearity and convergence error might be an issue especially with 
biological data. survival Fitters have a penalizer that can be called and adjusted to deal wit this issue, alternatively
you can try feature selection, dimensionality reduction or grouping variables into random/relevant groups.