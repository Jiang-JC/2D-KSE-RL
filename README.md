# 2D-KSE-RL

Repository for "Numerical solutions of fixed points in two-dimensional Kuramoto-Sivashinsky equation expedited by reinforcement learning".

## Requirements



## Installment
Create a new environment with Conda

```
conda env create -f KSE_RL.yaml
conda activate KSE_RL
```

## Repository Sturcture

```
Navigation_2DKSE                   // For task of navigating towards a fixed goal point
Navigation_2DKSE_Result            // Test effect of navigating towards a fixed goal point
Identification_2DKSE               // For task of dentifying fixed points
FixedPoints
    - FixedPointsData              // Data of fixed points in two-dimensional Kuramoto-Sivashinsky
    - FixedPointsImage             // Images of fixed points in two-dimensional Kuramoto-Sivashinsky
```

## Train



## Fixed points of 2D KSE


<img src="ImageForPresent\FixedPoints.png" width="1000">

**Figure 1.** Four exemplary fixed points in the 2D KSE in (a) phase space and (b) physical spatial domain. In the phase space representation, the figure is characterized by the set $\widehat{e}\_{(0,1)}$,  $\widehat{e}\_{(1,1)}$,  $\widehat{e}\_{(1,0)}$ in the Fourier space. For example, $\widehat{e}_{(0,1)}$ is the first Fourier mode in the y direction with zero Fourier mode in the x direction. In order to visually demonstrate the periodicity of the solution, we extend the results to $[0, 40] * [0, 40]$ in physical spatial domain.



<img src="ImageForPresent\FixedPointsList.png" width="1000">

**Table 1.** Fixed points list for 2D KSE. For conciseness, we tabulate only the absolute value of the first three complex-valued Fourier coefficients for each point. The absolute value has not been normalised by the spatial dimensions (64×64). Only one decimal place is retained for clarity. †The cases E1-E17 are further listed in table 2.



<img src="ImageForPresent\FixedPointsZeroList.png" width="1000">

**Table 2.** Representation of the fixed points E1-E17 in 2D KSE which all have the same $\widehat{e}\_{(0,1)} = \widehat{e}\_{(1,1)} = \widehat{e}\_{(1,0)} = 0$.


