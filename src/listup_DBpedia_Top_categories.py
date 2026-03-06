
"""
①-1. DBpediaから大カテゴリをリストアップするスクリプト
大カテゴリを手動で選別する必要があるため、中カテゴリのスクリプトとは分けている。

元となったコード：
* notebook20260110/listup_categories_and_get_properNouns_from_DBpedia.ipynb 内の⭐️部分
    * queryの試行錯誤はこのノートブック内で行った

uv run python src/listup_DBpedia_Top_categories.py
"""

import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)


DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
dbpedia_classes_path = os.path.join(project_root, "data", "dbpedia", "Top_classes.tsv")



query = """PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?class ?label
WHERE {
  GRAPH <http://dbpedia.org/resource/classes#> {
    ?class a owl:Class ;
           rdfs:subClassOf owl:Thing ;
           rdfs:label ?label .
    FILTER(lang(?label) = "en")
    FILTER(STRSTARTS(STR(?class), "http://dbpedia.org/ontology/"))
  }
}
ORDER BY LCASE(STR(?label)) STR(?class)
"""



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



def save_tsv(rows: List[Dict[str, Optional[str]]], out_path: Path) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["class_uri", "label"])
        for row in rows:
            writer.writerow([row["class_uri"], row["label"] or ""])



def main():
    # ⭐️ クエリを実行して結果を表示
    data = run_sparql(query)
    for b in data["results"]["bindings"][:20]:
        print(b["class"]["value"])

    # ⭐️ 結果をtsvに保存
    rows = []
    for b in data["results"]["bindings"]:
        rows.append({
            "class_uri": b["class"]["value"],
            "label": b["label"]["value"]
        })
    save_tsv(rows, dbpedia_classes_path)


if __name__ == "__main__":
    main()
    