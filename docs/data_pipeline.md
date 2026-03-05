# Data Pipeline（データ作成手順）

この文書は、学習/評価用データを再現可能に生成するための手順をまとめる。

## ① DBpediaからカテゴリをリストアップする
### ①-1. 大カテゴリのリストアップ
```sh
uv run python src/listup_DBpedia_Top_categories.py
```
-> `data/dbpedia/classes.tsv` が生成される。
生成されるファイル：
- `data/dbpedia/classes.tsv` : DBpediaの、Thingsクラス直下のカテゴリURIとラベルの一覧

### ①-2. 大カテゴリの選別
notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb 内の「実体を持つクラスを選ぶ」に当たる。

`data/dbpedia/classes.tsv` に、DBpediaのカテゴリURIとラベルの一覧が保存される。不要なカテゴリを手動で削除するなどして、学習/評価に使うカテゴリを絞り込み、
その結果を `data/dbpedia/classes.curated.tsv` として保存する。


### ①-3. 中カテゴリのリストアップ
```sh
uv run python src/listup_DBpedia_Mid_categories.py
```
生成されるファイル：
- `data/dbpedia/Top_to_Mid_classe_map.json` : 大カテゴリと中カテゴリの対応表
- `data/dbpedia/Mid_classes.json` : DBpediaの、中カテゴリURIとラベルの一覧（ただしsubclassが存在しなかった大カテゴリは、中カテゴリとして追加する(Medicineが該当)）

memo: Mid_classesは、元はdata/dbpedia_selected_subclasses.jsonだった。以降でdbpedia_selected_subclasses.jsonを参照している箇所は、Mid_classes.jsonに置き換える必要がある。
同様に、Top_to_Mid_classe_map.jsonも元はdata/dbpedia_classes_to_subclasses.jsonだったため、Top_to_Mid_classe_map.jsonに置き換える必要がある。

### ② wikiからカテゴリ(①)の固有名詞を大量にとってくる
```sh
uv run python src/listup_properNouns.py
```
生成されるファイル：
- data/dbpedia/wikidata_Things_childs_LIMIT1000/*.csv : 各中カテゴリに属する固有名詞リスト（固有名詞のQID, 固有名詞のラベル, カテゴリのラベル, カテゴリのQID）





