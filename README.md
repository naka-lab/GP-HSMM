# GP-HSMM

ガウス過程と隠れセミマルコフモデルを用いた時系列データの分節化の実装です．
ガウス過程の計算は，Cythonと計算のキャッシュを利用して高速化しています．
詳細は以下の論文を参照してください．

Tomoaki Nakamura, Takayuki Nagai, Daichi Mochihashi, Ichiro Kobayashi, Hideki Asoh and Masahide Kaneko, “Segmenting Continuous Motions with Hidden Semi-Markov Models and Gaussian Processes”, Frontiers in Neurorobotics, vol.11, article 67, pp. 1-11, Dec. 2017 [[PDF]](https://github.com/naka-lab/GP-HSMM/raw/master/main.pdf)

## 実行方法

```
python main.py
```

Cythonで書かれたプログラムは実行時に自動的にコンパイルされます．
WindowsのVisual Studioのコンパイラでエラーが出る場合は，

```
(Pythonのインストールディレクトリ)/Lib/distutils/msvc9compiler.py
```

の`get_build_version()`内の

```
majorVersion = int(s[:-2]) - 6
```

を使いたいVisual Studioのバージョンに書き換えてください．
VS2012の場合は，`majorVersion = 11`となります．


## 注意
GPSegmentation.pyのl.25-27で補助点を決めている．

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
