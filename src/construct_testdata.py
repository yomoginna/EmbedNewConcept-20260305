"""
test dataを構築するスクリプト

- target_conceptはtrain_data作成時に生成された、config/target_concepts.jsonの中のconceptとする
- 各target_conceptに対して、test_templateをもとに、正解選択肢と不正解選択肢を含むテストサンプルを作成する
- 正解選択肢は、target_conceptに属する概念の特徴から、rel_to_maskedSentenceのtemplateに当てはまるものを使用する
- 不正解選択肢は、target_conceptとは異なるカテゴリに属する概念の特徴から、rel_to_maskedSentenceのtemplateに当てはまるものをランダムに選択して使用する
- 作成したテストサンプルは、data/test_data/{target_concept}.json に保存する

```sh
uv run python src/construct_testdata.py
```

"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

from tqdm import tqdm

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from utils.wikipedia_api_utils import extract_wiki_main_text, fetch_wikipedia_page


num_incorrect_options = 2   # 学習データの1サンプルにおいて、正しい選択肢（正解）に加えて、誤った選択肢（不正解）をいくつ含めるか

test_template_path = os.path.join(project_root, 'data', 'templates', 'level1_2_test_template.txt')
rel_to_maskedSentence_path = os.path.join(project_root, 'data', 'templates', 'rel_to_maskedSentence.json')
target_category_to_concept_map_path = os.path.join(project_root, 'config', 'target_concepts.json')
train_data_dir = os.path.join(project_root, 'data', 'train_data')
generated_facts_dir = os.path.join(project_root, 'data', 'generated_facts_in_wiki')
test_samples_save_dir = os.path.join(project_root, 'data', 'test_data_tmp')
os.makedirs(test_samples_save_dir, exist_ok=True)


def main():

    with open(test_template_path, "r") as f:
        test_template = f.read()

    with open(rel_to_maskedSentence_path, 'r') as f:
        rel_to_maskedSentence = json.load(f)

    with open(target_category_to_concept_map_path, 'r') as f:
        target_category_to_concept_map = json.load(f)


    for category, target_concepts in target_category_to_concept_map.items():

        # ****** 不正解選択肢用のデータとして、他のcategoryに属する概念の特徴を関係毎に取得 ******
        rel_to_other_category_feats = defaultdict(list) # key: rel, value: 他のcategoryの概念の特徴のリスト
        for cat, concepts in target_category_to_concept_map.items():
            if cat == category:
                # 他のcategoryの概念を不正解選択肢用のデータとして使用するため、同じcategoryの概念はスキップ
                continue
            for concept in concepts:
                # print(f"Concept: {concept}")
                # 生成した特徴のファイルを読み込む
                with open(os.path.join(generated_facts_dir, f"{concept.replace(' ', '_')}.json"), 'r') as f:
                    incorrect_candidate_option_facts = json.load(f)
                
                # rel_to_maskedSentence をもとに、factsの各特徴文から[MASK]に該当する部分を抜き取る
                for rel, facts in incorrect_candidate_option_facts['parsed']['english'].items():
                    # print(f"\tRelation: {rel}")
                    template = rel_to_maskedSentence.get(rel)
                    mask_pattern = re.escape(template).replace("\\[MASK\\]", "(.+)")  # [MASK]をキャプチャグループに置換 (templateは常に.で終わらせているため、!?になっていたり、.なしの文はマッチしない)
                    for fact in facts:
                        if fact.lower() in ["unknown", '不明']:
                            continue

                        match = re.match(mask_pattern, fact)    # templateの[MASK]を(.+)に置換した正規表現パターンを作成
                        if match:
                            maskFeat = match.group(1)  # キャプチャグループから[MASK]部分を抽出
                            rel_to_other_category_feats[rel].append(maskFeat)
                        else:
                            # 特徴文がtemplateの形式でなかった場合:
                            # print(f"no match Fact: {rel} - '{fact}'")
                            pass


        # ****** 正解選択肢用のデータとして、target_conceptsに属する概念の特徴を関係毎に取得 ******
        for target_concept in target_concepts:
            random.seed(42)  # 不正解選択肢のランダム選択の再現性のため、シードを固定 (指定したtarget_conceptの順序等が変わっても、同じ不正解選択肢が選ばれるようにするために、ここでシード固定)

            # 学習データにした特徴のファイルを読み込む
            with open(os.path.join(train_data_dir, f"{target_concept.replace(' ', '_')}.json"), 'r') as f:
                data = json.load(f)
            facts = data['facts']

            # rel_to_maskedSentence をもとに、factsの各特徴文から[MASK]に該当する部分を抜き取る
            rel_to_maskFeats = {}
            for fact in facts:
                for rel, template in rel_to_maskedSentence.items():
                    # factがそのrelのtemplateに当てはまるか
                    # templateの[MASK]を(.+)に置換した正規表現パターンを作成
                    mask_pattern = re.escape(template).replace("\\[MASK\\]", "(.+)")  # [MASK]をキャプチャグループに置換
                    match = re.match(mask_pattern, fact)
                    if match:
                        maskFeat = match.group(1)  # キャプチャグループから[MASK]部分を抽出
                        if rel not in rel_to_maskFeats:
                            rel_to_maskFeats[rel] = []
                        rel_to_maskFeats[rel].append(maskFeat)
                        break
                else:
                    # break されなかったときだけ実行される
                    print(f"no match Fact: '{fact}'")
            print(f"?{len(facts)} == {sum(len(feats) for feats in rel_to_maskFeats.values())}")

            # *** concept毎にtest sampleを作成し、保存する ***
            test_samples = []
            test_id = 0
            for rel, mask_feats in rel_to_maskFeats.items():
                print(f"Relation: {rel}, Masked Features: {mask_feats[:5]}")
                for correct_option_maskFeat in mask_feats:

                    if len(rel_to_other_category_feats.get(rel, [])) < num_incorrect_options:
                        # 不正解選択肢用の特徴が足りない場合は、そのrelに関するサンプルの作成をスキップ
                        print(f"Not enough incorrect options for relation '{rel}'. Skipping this sample.")
                        continue
                    
                    # 不正解選択肢をランダムに選択
                    incorrect_option_maskFeats = random.sample(rel_to_other_category_feats.get(rel, ["Unknown"]), num_incorrect_options) # relに対応する不正解選択肢がない場合は "Unknown" を使用
                    if any(f.lower() in ["unknown", "不明"] for f in incorrect_option_maskFeats):
                        continue


                    # *** test_id に応じて、正解選択肢番号を 1->2->...->(num_incorrect_options+1)->1->2->... の順で割り当てる ***
                    num_options = num_incorrect_options + 1
                    correct_option_num = (test_id % num_options) + 1
                    options = [''] * num_options
                    # 正解選択肢を正解選択肢番号の位置に配置し、残りの位置に不正解選択肢を配置する
                    options[correct_option_num - 1] = correct_option_maskFeat
                    incorrect_idx = 0
                    for i in range(num_options):
                        if i == correct_option_num - 1:
                            continue
                        options[i] = incorrect_option_maskFeats[incorrect_idx]
                        incorrect_idx += 1
                    
                    # optionsを "1. option1\n2. option2\n..." の形式の文字列に変換
                    options = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

                    # test_templateの[SUMMARY]部分にsummaryを入れ、[FACT]部分にrel_to_maskedSentenceのrelに対応するmasked sentenceを入れたものを作成
                    test_sample = test_template.format(
                        sentence=rel_to_maskedSentence[rel].replace("It ", "<target_token> "),
                        options=options
                    )
                    test_samples.append({
                        'test_id': test_id,
                        'relation': rel,
                        'correct_feat': correct_option_maskFeat,
                        'options': options,
                        'correct_num': correct_option_num,
                        'test1': test_sample,
                    })
                    test_id += 1

            with open(os.path.join(test_samples_save_dir, f"{target_concept.replace(' ', '_')}.json"), "w") as f:
                json.dump(test_samples, f, ensure_ascii=False, indent=4)
        

if __name__ == "__main__":
    main()

    
