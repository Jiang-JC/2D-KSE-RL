# 2D-KSE-RL

Repository for "Numerical solutions of fixed points in two-dimensional Kuramoto-Sivashinsky equation expedited by reinforcement learning" by Juncheng Jiang, Dongdong Wan and Mengqi Zhang†. ([website][https://royalsocietypublishing.org/doi/full/10.1098/rspa.2024.0222])

†Corresponding authors

We provided the codes for Fixed point navigation, Identification and Linear stability analysis.

If you use the fixed point data and codes from this repo, please cite our work. Thanks!
```
@inproceedings{jiang2025numerical,
  title={Numerical solutions of fixed points in two-dimensional Kuramoto--Stravinsky equation expedited by reinforcement learning},
  author={Jiang, Juncheng and Wan, Dongdong and Zhang, Mengqi},
  booktitle={Proceedings A},
  volume={481},
  number={2305},
  pages={20240222},
  year={2025},
  organization={The Royal Society}
}
```

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
FixedPoints                        // Data of fixed points
Linear_Stability_Analysis          // Linear stability analysis for fixed points
```

## Train
Select the "Navigation_2DKSE" folder for the task of navigating towards a fixed goal point, or select the "Identification_2DKSE" folder for the task of identifying fixed points, then

```
python main.py
```
This code was originally designed for execution on HPC, allowing for a large replay buffer size. For execution on a personal computer, it is advisable to reduce the replay buffer size, for example, by a factor of 10, to accommodate limited memory resources. The replay buffer parameter can be adjusted in the buffer.py file.

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

## Linear Stability Analysis
We employed two methods to confirm the linear instability of the 303 fixed points. Calculations show that all the 303 fixed points obtained are linearly unstable to infinitesimal perturbations. Detailed data, examples, methodologies, and instructions for using the data can be found in the "Linear_Stability_Analysis directory".

