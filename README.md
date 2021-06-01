# PEPSDI  

PEPSDI (**P**articles **E**ngine for **P**opulation **S**tochastic **D**ynam**I**cs) is a Bayesian inference framework for dynamic state space mixed effects models (SSMEM).  A SSMEM is a powerful tool for modelling multi-individual data since it can account for intrinsic (intra-individual), extrinsic (inter-individual) and residual (measurement-error) variability. Overall, this makes SSMEM:s suitable for, and not limited to,  problems in pharmacology [1] and cellular biology [2]. 

In contrast to non-linear mixed effects frameworks, such as Monolix [3], PEPSDI assumes an underlaying stochastic dynamic model. PEPSDI currently supports stochastic differential equations (SDE:s). For chemical reaction networks (e.g a cellular pathway) PEPSDI further supports the SSA (Gillespie) [4], Extrande [5], and tau-leaping [6] simulators.  

 
By levering Hamiltonian-Monte-Carlo sampling via the [Turing-library](https://github.com/TuringLang/Turing.jl) PEPSDI is flexible regarding the probability distribution of the random-effects.  Currently, the random effects can be modelled via a log-normal distribution with either full, or diagonal, covariance matrix. However, additional parameterisations (distributions) can be implemented.  

 
PEPSDI is introduced, and described, in the manuscript *add*. To get started with PEPSDI the examples in the manuscript are available as notebooks under *Code/Examples*. This folder also contains notebooks on how to leverage the underlaying algorithms in PEPSDI to perform single-individual inference. An extensive tutorial is further available under the examples-folder. The results presented in the manuscript were produced by the scripts in *Code/Result_paper*-folder.  

This repository is a first (pre-alpha) version of PEPSDI. If wanting to use PEPSDI, and need help, the corresponding author in the manuscript can be contacted.  

## A closer look at PEPSDI  

PEPSDI performs full Bayesian inference for SSMEM. This means that PEPSDI infers individual parameters *c<sub>i*, between individual constant parameters *ĸ*, strength of measurement error *ξ* and population parameters η; *c<sub>i* ~ *π(c<sub>i* *| η)*. This is achieved by using a Gibbs-sampler to target the full-posterior. Some Gibbs-steps have a tractable likelihood and are thus sampled via HMC, while the remaining Gibbs-step have an intractable likelihood. These are sampled via pseduo-marginal inference [7] where particle-filters are employed to obtain an unbiased likelihood estimate [8]. To properly tune the particles filters, as shown in the example notebooks, a pilot run is required. Furthermore, for computational efficiency we employ, when possible, correlated-particle filters [9], and tune the parameters proposal distributions using adaptive algorithms [10]. 

PEPSDI can be run with two options. We recommend the second (default in notebooks) where constant parameters (*ĸ*, *ξ*) are allowed to vary weakly between cells. This can speed up inference by more than a factor 30.  

## Requirements for reproducing the result 

PEPSDI was developed using Julia 1.5.2 on Linux (has been successfully run on both Fedora and Ubunutu). PEPSDI should work with newer versions off Julia, especially if the toml-file is used to create an environment with all correct dependencies.  

Since PEPSDI was developed on Linux, there might be problems with the file-path on Windows. One way to potentially resolve this for a Windows user might be (I have not tested this) to download the Ubuntu-terminal for Windows 

## References

1. Donnet S, Samson A.  A review on estimation of stochastic differential equationsfor pharmacokinetic/pharmacodynamic models.  Advanced Drug Delivery Reviews. 2013jun;65(7):929–939
2. Zechner C, Unger M, Pelet S, Peter M, Koeppl H.  Scalable inference of heterogeneousreaction kinetics from pooled single-cell recordings. Nature Methods. 2014 feb;11(2):197–202
3. Monolix version 2019R2. Antony, France: Lixoft SAS; 2019. http://lixoft.com/products/monolix/.
4. Gillespie DT. Exact stochastic simulation of coupled chemical reactions. In: Journal ofPhysical Chemistry. vol. 81. American Chemical Society; 1977. p. 2340–2361.
5. Voliotis M, Thomas P, Grima R, Bowsher CG. Stochastic Simulation of Biomolecular Net-works in Dynamic Environments. PLOS Computational Biology. 2016 jun;12(6):e1004923.
6. Gillespie DT.   Chemical Langevin equation.   Journal of Chemical Physics. 2000jul;113(1):297–306.
7. Andrieu C, Doucet A, Holenstein R. Particle Markov chain Monte Carlo methods. Journalof the Royal Statistical Society: Series B (Statistical Methodology). 2010 jun;72(3):269–342.
8. Pitt MK, Silva RDS, Giordani P, Kohn R. On some properties of Markov chain MonteCarlo simulation methods based on the particle filter. In: Journal of Econometrics. vol.171. North-Holland; 2012. p. 134–151.
9. Deligiannidis G, Doucet A, Pitt MK. The Correlated Pseudo-Marginal Method. Journalof the Royal Statistical Society Series B: Statistical Methodology. 2018 nov;80(5):839–870.
10. Andrieu C, Thoms J. A tutorial on adaptive MCMC. Statistics and Computing. 2008dec;18(4):343–37