# GP-HSMM

ガウス過程と隠れセミマルコフモデルを用いた時系列データの分節化の実装です．詳細は以下の論文を参照してください．

Tomoaki Nakamura, Takayuki Nagai, Daichi Mochihashi, Ichiro Kobayashi, Hideki Asoh and Masahide Kaneko, “Segmenting Continuous Motions with Hidden Semi-Markov Models and Gaussian Processes”, Frontiers in Neurorobotics, vol.11, article 67, pp. 1-11, Dec. 2017 [[PDF]](https://github.com/naka-lab/GP-HSMM/raw/master/main.pdf)

さらに以下の文献で提案された高速化法を導入，計算のCython化，逆行列演算の工夫により，従来のGP-HSMMに比べ高速な計算が可能です．

川村 美帆，佐々木 雄一，中村 裕一，"GP-HSMM の尤度計算並列化による高速な身体動作の分節化方式"，計測自動制御学会 システムインテグレーション部門講演会，1A4-08，2021

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
