# GP-HSMM

This is an implementation of time series data segmentation using Gaussian Processes (GP) and Hidden Semi-Markov Models (HSMM). For details, please refer to the following paper:

Tomoaki Nakamura, Takayuki Nagai, Daichi Mochihashi, Ichiro Kobayashi, Hideki Asoh and Masahide Kaneko, “Segmenting Continuous Motions with Hidden Semi-Markov Models and Gaussian Processes”, Frontiers in Neurorobotics, vol.11, article 67, pp. 1-11, Dec. 2017 [[PDF]](https://github.com/naka-lab/GP-HSMM/raw/master/main.pdf)

**A fast and scalable implementation called [RFF-GP-HSMM](https://github.com/naka-lab/RFF-GP-HSMM), which solves the slow computation problem of GP-HSMM, is also available.**


## How to Run

```
python main.py
```

Programs written in Cython will be automatically compiled at runtime.
If you encounter compilation errors with the Visual Studio compiler on Windows, please edit:

```
(Python installation directory)/Lib/distutils/msvc9compiler.py
```

Inside the `get_build_version()` function, replace the following line:

```
majorVersion = int(s[:-2]) - 6
```

with the version number of the Visual Studio you wish to use.
For example, for VS2012, set:

```
majorVersion = 11
```

## Output Files

When executed, the following files and directories will be created in the specified folder:

| File Name| Description |
| ---- | --- |
| class{c}.npy         | A collection of segments classified into class c                                                              |
| class{c}\_dim{d}.png | Plot of the d-th dimension of segments classified into class c                                                |
| segm{n}.txt          | Segmentation result of the n-th sequence. Column 1: segment class, Column 2: flag indicating segment boundary |
| trans\_bos.npy       | Probability that each class appears at the beginning of a sequence                                            |
| trans\_eos.npy       | Probability that each class appears at the end of a sequence                                                  |
| trans.npy            | Transition probabilities of each class appearing after a given class                                          |


# LICENSE
This program is freely available for free non-commercial use. 
If you publish results obtained using this program, please cite:

```
@article{nakamura2017segmenting,
  title={Segmenting continuous motions with hidden semi-markov models and gaussian processes},
  author={Nakamura, Tomoaki and Nagai, Takayuki and Mochihashi, Daichi and Kobayashi, Ichiro and Asoh, Hideki and Kaneko, Masahide},
  journal={Frontiers in neurorobotics},
  volume={11},
  pages={67},
  year={2017},
  publisher={Frontiers}
}
```
