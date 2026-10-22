# WRIMEによる国会発言の感情分析（200件実験）

WRIMEで学習済みの公開モデルを用いて、国会発言の8感情と感情極性を推定する小規模実験です。

## 1. 200件を抽出

8トピックから各25件を、固定乱数（seed=42）で抽出します。

```bash
python3 prepare_sample.py
```

出力: `outputs/speeches_sample_200.csv`

## 2. 環境構築

Python 3.12を推奨します。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 3. 感情分析

```bash
.venv/bin/python analyze_wrime.py
```

出力: `outputs/speeches_sample_200_wrime.csv`

`wrime_joy`〜`wrime_trust`は各感情が表れている確率（感情間の合計は1になりません）、`wrime_sentiment`は-1（否定的）〜+1（肯定的）の極性です。長い発言は文境界で分割し、区間スコアを平均します。これは探索的分析であり、話者の真の内面を推定するものではありません。

## 使用モデル

- `Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime`
- `katsutaku/wrime-sentiment-analyzer`

いずれもWRIMEを用いて学習された公開モデルです。WRIME自体はCC BY-NC-ND 4.0のため、利用・成果公開時は原データセットの条件を確認してください。

## 4. 集計と検査

```bash
.venv/bin/python summarize_results.py
```

値域を検査し、`outputs/wrime_summary_by_cluster.csv` にトピック別平均を出力します。

## 5. PNGヒートマップ

```bash
.venv/bin/python create_heatmap_png.py
```

出力:

- `outputs/wrime_emotions_heatmap.png`
- `outputs/wrime_sentiment_heatmap.png`

個別極性スコアの分布:

```bash
.venv/bin/python create_sentiment_distribution_png.py
```

出力: `outputs/wrime_sentiment_distribution.png`

8クラスタ別の箱ひげ図:

```bash
.venv/bin/python create_cluster_boxplot_png.py
```

出力: `outputs/wrime_sentiment_boxplot_by_cluster.png`

8クラスタ別のバイオリンプロット:

```bash
.venv/bin/python create_cluster_violinplot_png.py
```

出力: `outputs/wrime_sentiment_violinplot_by_cluster.png`

8クラスタ間のKruskal–Wallis検定:

```bash
.venv/bin/python test_cluster_polarity.py
```

出力: `outputs/kruskal_wallis_polarity.md`

50件のAI参照ラベル作成と暫定評価:

```bash
python3 prepare_annotation_sample.py
python3 apply_reference_labels.py
.venv/bin/python evaluate_reference_labels.py
```

出力:

- `outputs/annotation_sample_50_labeled.csv`
- `outputs/evaluation_50_ai_reference.md`
