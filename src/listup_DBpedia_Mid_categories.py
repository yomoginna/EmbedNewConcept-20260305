
"""
①-3. DBpediaからカテゴリをリストアップするスクリプト
元となったコード：
* notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb 内の⭐️部分
    * queryの試行錯誤はこのノートブック内で行った

_preからの変更点:
* 最初から、直下だけでなく、その子の子の子...も含めた全てのサブクラスを取得するようにした（SPARQL_MULTIHOP_GET_SUBCLASSES_BASEを使用）。
    理由は、直下のサブクラスだけだと、Mid levelカテゴリ数が不十分だったため。


uv run python src/listup_DBpedia_Mid_categories.py > output.log 2>&1
nohup uv run python src/listup_DBpedia_Mid_categories.py > output.log 2>&1 & # 3322556
"""

import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import requests

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)


# ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
dbpedia_classes_path = os.path.join(project_root, "data", "dbpedia", "Top_classes.tsv")
dbpedia_mid_classes_path = os.path.join(project_root, "data", "dbpedia", "Top_to_Mid_classe_map.json")
dbpedia_classes_flatten_path = os.path.join(project_root, "data", "dbpedia", "Mid_classes.json") # 元はdata/dbpedia_selected_subclasses.jsonだった

SUBCLASS_NUM_THRESHOLD = 10 # 直下のサブクラスがこれ以上あるカテゴリをMid levelカテゴリとする。これを下回る場合は、Mid levelカテゴリとしては不十分とみなす


def run_sparql(query: str, timeout: int = 30) -> dict:
    headers = {
        # できれば連絡先つきのUAにする（エンドポイントに優しい）
        "User-Agent": "my-research-bot/0.1 (contact: you@example.com)",
        "Accept": "application/sparql-results+json",
    }
    params = {
        "query": query,
        "format": "json",  # DBpediaはこれでOK
    }
    r = requests.get(DBPEDIA_SPARQL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()



SPARQL_MULTIHOP_GET_SUBCLASSES_BASE = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subClass ?label
WHERE {{
    ?subClass rdfs:subClassOf* <{category}> .
    ?subClass rdfs:label ?label .
    FILTER(lang(?label) = "en")
}}
ORDER BY LCASE(STR(?label))
"""



def main():

    with open(dbpedia_classes_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        classes = [row for row in reader]

    categories = [c["class_uri"] for c in classes]
    print(f"Total categories: {len(categories)}: {categories[:5]} ... {categories[-5:]}")


    # *** Mid levelのカテゴリとして、各Top level カテゴリのdboサブクラスを取得する ***

    # memo: なぜか途中で止まることがあるので、途中から実行再開できるよう、結果を随時保存する
    if os.path.exists(dbpedia_mid_classes_path):
        with open(dbpedia_mid_classes_path, 'r') as f:
            result = json.load(f)
        print(f"result already has {len(result)} ({list(result.keys())}) categories with subclasses. Will fetch subclasses for remaining {len(categories) - len(result)} categories.")

    else:
        # 各カテゴリの直下のdboサブクラスを取得するクエリを生成
        result: Dict[str, List[Tuple[str, str]]] = {}


    for category in categories:
        print(category)
        # # 既に結果がある場合:
        # if category in result.keys():
        #     # このカテゴリにsubclassがSUBCLASS_NUM_THRESHOLD以上あれば、すでに十分なsubclassを取得できているとみなしてスキップ。そうでなければ、subclassがSUBCLASS_NUM_THRESHOLD未満なので、再度クエリを実行してsubclassを取得し直す。
        #     if len(result[category]) >= SUBCLASS_NUM_THRESHOLD:
        #         print(f"\tAlready have result for {category}, skipping...")
        #         continue

        # *** サブクラスを再帰的に取得する ***
        query = SPARQL_MULTIHOP_GET_SUBCLASSES_BASE.format(category=category)

        data = run_sparql(query)
        bindings = data.get("results", {}).get("bindings", [])

        rows = []
        if bindings:
            for b in bindings:
                subClass_uri = b["subClass"]["value"]
                name = b["label"]["value"]
                rows.append((subClass_uri, name))

        
        if len(rows) == 0:
            print(f"\tNo subclasses found for {category} even with multi-hop query, skipping saving in new file...")
            continue
        
        # *** 1-hop + multi-hopの結果を、class - subclasses の構造にして随時保存 ***
        result[category] = rows
        print(f"{category}: {len(rows)} direct subclasses")
        with open(dbpedia_mid_classes_path, 'w') as f:
            json.dump(result, f, indent=2)


    

    # categories と，categories直下のsubclasses (result) を結合して保存
    all_classes = []
    for category in categories:
        if category in result.keys():
            # そのcategoryのsubClassesを取得できていれば，subClassesを追加. そのcategory自体は追加しない
            all_classes.extend(result[category])
        else:
            # そのcategoryのsubClassesがなかったのであれば，category自体を追加
            name = category.split("/")[-1]
            all_classes.append((category, name))
    print(len(all_classes), all_classes)

    # 保存
    with open(dbpedia_classes_flatten_path, "w") as f:
        json.dump(all_classes, f, indent=2)


if __name__ == "__main__":
    main()