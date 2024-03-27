# 2D-KSE-RL

Repository for "Numerical solutions of fixed points in two-dimensional Kuramoto-Sivashinsky equation expedited by reinforcement learning" by Juncheng Jiang, Dongdong Wan and Mengqi Zhang†.

†Corresponding authors

This paper will be submitted for publication.

## Dependencies
+ Python  3.9
+ Pytorch  2.2.1
+ Numpy
+ Matplotlib
+ imageio

## Repository Sturcture

The following gives a brief overview of the contents.

```
Navigation_2DKSE                   // For the task of navigating towards a fixed goal point
Navigation_2DKSE_Result            // Test the effect of navigating towards a fixed goal point
Identification_2DKSE               // For the task of dentifying fixed points
FixedPoints
    - FixedPointsData              // Data of fixed points in two-dimensional Kuramoto-Sivashinsky
    - FixedPointsImage             // Images of fixed points in two-dimensional Kuramoto-Sivashinsky
```

## Train
Select the "Navigation_2DKSE" folder for the task of navigating towards a fixed goal point, or select the "Identification_2DKSE" folder for the task of identifying fixed points, then

```
python main.py
```

## Test
For the task of navigating towards a fixed goal point, Put the trained model into the "Navigation_2DKSE_Result\Models" folder, then

```
python mainPhaseDomain.py
python mainSpatialDomain.py
```

For the task of identifying fixed points, after training the model, you can find fixed points in folder "Identification_2DKSE\FixedPoints"

## Fixed points of 2D KSE

<img src="ImageForPresent\FixedPoints.png" width="1000">

**Figure 1.** Four exemplary fixed points in the 2D KSE in (a) phase space and (b) physical spatial domain. In the phase space representation, the figure is characterized by the set $\widehat{e}\_{(0,1)}$,  $\widehat{e}\_{(1,1)}$,  $\widehat{e}\_{(1,0)}$ in the Fourier space. For example, $\widehat{e}_{(0,1)}$ is the first Fourier mode in the y direction with zero Fourier mode in the x direction. In order to visually demonstrate the periodicity of the solution, we extend the results to $[0, 40] * [0, 40]$ in physical spatial domain.

