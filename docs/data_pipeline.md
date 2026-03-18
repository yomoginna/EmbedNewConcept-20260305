# Data Pipeline（データ作成手順）

この文書は、学習/評価用データを再現可能に生成するための手順をまとめる。

## ① DBpediaからカテゴリをリストアップする
### ①-1. 大カテゴリのリストアップ
```sh
uv run python src/listup_DBpedia_Top_categories.py
```
-> `data/dbpedia/Top_classes.tsv` が生成される。
生成されるファイル：
- `data/dbpedia/Top_classes.tsv` : DBpediaの、Thingsクラス直下のカテゴリURIとラベルの一覧

### ①-2. 大カテゴリの選別
notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb 内の「実体を持つクラスを選ぶ」に当たる。

`data/dbpedia/Top_classes.tsv` に、DBpediaのカテゴリURIとラベルの一覧が保存される。不要なカテゴリを手動で削除するなどして、学習/評価に使うカテゴリを絞り込み、
その結果を `data/dbpedia/Top_classes.curated.tsv` として保存する。


### ①-3. 中カテゴリのリストアップ
```sh
uv run python src/listup_DBpedia_Mid_categories.py
```
生成されるファイル：
- `data/dbpedia/Top_to_Mid_classe_map.json` : 大カテゴリと中カテゴリの対応表
- `data/dbpedia/Mid_classes.json` : DBpediaの、中カテゴリURIとラベルの一覧（ただしsubclassが存在しなかった大カテゴリは、中カテゴリとして追加する(Medicineが該当)）

memo: Mid_classesは、元はdata/dbpedia_selected_subclasses.jsonだった。以降でdbpedia_selected_subclasses.jsonを参照している箇所は、Mid_classes.jsonに置き換える必要がある。
同様に、Top_to_Mid_classe_map.jsonも元はdata/dbpedia_classes_to_subclasses.jsonだったため、Top_to_Mid_classe_map.jsonに置き換える必要がある。

## ② DBpediaからカテゴリに属する固有名詞をリストアップする
### ②-1. wikiからカテゴリ(①)の固有名詞を大量にとってくる
```sh
uv run python src/listup_properNouns.py
```
生成されるファイル：
- data/dbpedia/wikidata_Things_childs_LIMIT1000/*.csv : 各中カテゴリに属する固有名詞リスト（固有名詞のQID, 固有名詞のラベル, カテゴリのラベル, カテゴリのQID）

[WIP] ②-2は無いかも
[WIP] ①-3を改良し(2026/03/05)、subclassを、class直下だけでなくその子も再帰的に取るように修正したため、subclassに属す固有名詞同士が重複する可能性がある。
    その場合は、初期化用平均vec計算には重複した固有名詞を利用し、新規概念の元になる固有名詞としては使わないことにする。


--- ここまでは生成済み ---

## ③ Wikipediaから、カテゴリに属する固有名詞の特徴文を生成する
### ③-1. wikiから特徴文を生成する

特定のカテゴリに対して生成する場合:
```sh
nohup uv run python src/gen_features_from_wiki.py --target_categories "board game" "Painting"  > output.log 2>&1 & # "Painting"
```
2865016

全カテゴリに対して生成する場合:
```sh
uv run python src/gen_features_from_wiki.py
```
生成されるファイル：
- data/generated_facts_in_wiki/*.json : 各固有名詞の特徴文と、その時の生成情報が全て、各concept名をファイル名としたjson形式で保存される.

現状:
- Painting カテゴリのみ、以下の条件で生成済み.
    model = "gemini-2.5-flash-lite"
    max_retries = 1
    feat_num_threshold = 60 # 1概念あたりで、生成された特徴の数がこの数以上であれば生成成功とみなす
    wiki_word_num_threshold = 500 # 十分な情報量のあるwikipageに対してのみ生成を行うための、wikipageの本文の単語数の閾値
    propnoun_num_for_new_concept = 50
    propnoun_num_for_init_vec = 100
    temperature = 0.2
    topP = 0.8



## ④ 特徴文ペアから、固有名詞を推定する→推定がうまくいかないのでskip (言い換えや、類似語句が予測されてしまい、推定に有用な特徴も多く不正解になってしまうため)
### ④-1. 特徴文ペアから、固有名詞を推定する
```sh
uv run python src/gen_guess_proper_noun_from_sentence_pair.py > output.log 2>&1
```
### ④-1. 十分に情報量のある特徴文を抽出する
n(=2)回以上正解した特徴文を選出
select_useful_features_based_on_genai_guess_concept_results.py を実行したいが、まだ修正していない


## ⑤ 特徴文をもとに、学習/評価用のデータセットを作成する
### ⑤-1. 学習用のデータセットを作成する
trainMemVec_fromXvec_gemma_wholeRun.py内で、ここで作成したファイルを元に、main_text or summaryと、factsから複数サンプルの学習データを構築することになる。

train data作成対象となるconceptを指定する場合:
```sh
uv run python src/construct_traindata.py --target_concepts "Adoration of the Kings" "Unlock!" "BattleFleet Mars"
```

wikiから抽出された特徴が一定以上存在するconceptすべてをtrain data作成対象とする場合:
```sh
uv run python src/construct_traindata.py
```


