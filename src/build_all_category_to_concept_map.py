"""
全midカテゴリについて、属す固有名詞リストを作成し、 mid category - proper noun list のmapを作成して保存する。

```sh 
uv run python src/build_all_category_to_concept_map.py
```
"""
import json
import os
import sys
# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)
from utils.handle_data_from_dbpedia_utils import loadProperNounData

propnoun_num_for_init_vec=100   # 初期化vecの作成に使う固有名詞の最低数. 例えば100に設定した場合、各カテゴリで最低100個の固有名詞を使用して初期化vecを作成することになる。(実際には、新規概念用にならなかった固有名詞全て使用する)
propnoun_num_for_new_concept=30 # 新規概念の元にする概念の作成に使う固有名詞の数. 例えば30に設定した場合、各カテゴリで30個の固有名詞を使用して新規概念の元にする概念の作成に使用することになる。

concept_map_path = os.path.join(project_root, "config", "all_concepts.json")


def main():
    # ****** Mid class (category) - target concepts のmapを作成して保存する ******
    # 全てのカテゴリ・固有名詞リスト の辞書を読み込む (重複等のfiltering済み)
    filtered_category_properNouns_dict = loadProperNounData(
        propnoun_num_threshold = propnoun_num_for_init_vec + propnoun_num_for_new_concept,
        print_flag=True
    )

    # 保存
    with open(concept_map_path, "w") as f:
        json.dump(filtered_category_properNouns_dict, f, ensure_ascii=False, indent=4)

    print("Training data construction completed.")
    

if __name__ == "__main__":
    main()
        
