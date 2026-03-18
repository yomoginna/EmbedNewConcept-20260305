"""
uv run python src/gen_guess_proper_noun_from_sentence_pair.py > output.log 2>&1
"""
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import time
import random

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
load_dotenv() # API keyの読み込み

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
sys.path.append(project_root)

from utils.llm_generation_utils import gen_with_google_genai_api

# from openai import OpenAI
# client = OpenAI()

# gptは少し高いのでgeminiに変更。wikiからの特徴抽出タスクは複雑な思考を必要としないため、軽量のモデルで十分と判断。
from google import genai
client = genai.Client()

# 定数
model = "gemini-2.5-flash-lite"
max_retries = 1
temperature = 0.2
topP = 0.8
delay_between_requests = 1  # APIリクエスト間の遅延（秒）
RANDOM_SEED = 42

def main():

    # 生成済みの, wikipageを元にしたconcept特徴文の読み込み先
    feat_sentences_dir = os.path.join(project_root, "data", "generated_facts_in_wiki")
    # gen_target_conceptfiles = os.listdir(feat_sentences_dir)

    # 生成対象のconceptを手動で指定する場合:
    target_concepts = ['Unlock!'] # ['BattleFleet Mars'] # ['Adoration of the Kings']
    gen_target_conceptfiles = [f"{concept.replace(' ', '_')}.json" for concept in target_concepts]

    # prompt と jsonschema を読み込み
    prompt_path = os.path.join(project_root, "data", "prompt", "predict_proper_noun_from_2sentences_baseprompt.txt")
    schema_path = os.path.join(project_root, "data", "prompt", "predict_proper_noun_from_2sentences_schema.json")
    with open(prompt_path, "r", encoding="utf-8") as f:
        gen_guess_concept_prompt_base = f.read()
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # 保存先準備
    generated_savedir = os.path.join(project_root, "data", "generated_guess_proper_noun_from_2facts")
    os.makedirs(generated_savedir, exist_ok=True)
    
    # ****** 各conceptの特徴文から固有名詞推定を実行 ******
    print("Generating guessed proper nouns from sentence pairs...")
    # 以下色々loop方法をコメントアウトしているのは，生成に時間がかかりすぎるので分割して実行したため．本来であれば引数+pythonのthreadで指定できると良かったが面倒なので手動で分割．
    
    for filename in tqdm(gen_target_conceptfiles, desc="Concepts"): # 手動で指定した場合
    # 順方向に読み込む(基本)
    # for filename in tqdm(os.listdir(feat_sentences_dir), desc="Concepts"):
    # 逆方向に読み込む
    # for filename in tqdm(os.listdir(feat_sentences_dir)[::-1], desc="Concepts"):

    # 5個ずつ分割実行用:
    # for filename in tqdm(os.listdir(feat_sentences_dir)[:5], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[5:10], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[10:15], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[15:20], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[20:25], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[25:30], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[30:35], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[35:40], desc="Concepts"):
    # for filename in tqdm(os.listdir(feat_sentences_dir)[40:180], desc="Concepts"): -> finished

        target_concept = filename.split('.json')[0].replace('_', ' ')  # Remove .txt and replace underscores with spaces
        print(target_concept)

        # *** 🔸特徴判定を生成済みのobjなら生成したデータを読み込んでおく (生成済みのcombを個別にskipするため) ***
        try:
            with open(os.path.join(generated_savedir, f"{target_concept.replace(' ', '_')}.jsonl"), "r", encoding="utf-8") as f:
                identify_gen_dt = [json.loads(line) for line in f.readlines()]
        except FileNotFoundError:
            identify_gen_dt = None


        # *** wikipageから生成した特徴文を読み込む ***
        with open(os.path.join(feat_sentences_dir, filename), "r", encoding="utf-8") as f:
            gen_info = json.load(f)
        gen_rel_to_facts = gen_info['parsed']['english']

        # *** wikipageから生成した特徴文(english)に, 通し番号を付ける ***
        id = 0
        dt_with_id = {}
        id_to_sentence = {}
        rel_to_ids = {}
        for rel, facts in gen_rel_to_facts.items():
            dt_with_id[rel] = []
            for s in facts:
                # dt_with_id[rel].append({
                dt_with_id.setdefault(rel, []).append({
                    "id": id,
                    "sentence": s
                })
                id_to_sentence[id] = s
                rel_to_ids.setdefault(rel, []).append(id)
                id += 1

        # ****** 特徴文ペアに対して固有名詞推定を実行 ******
        # 全通りの特徴文ペア作成する場合:
        # combs, id_to_sentence_filtered = pairwise_sentences_combinations(id_to_sentence)
        
        # アンカーとなる定義文を、ランダムにn個選び、それと他の特徴文をペアにして生成する場合:
        combs, id_to_sentence_filtered = pairwise_sentence_combinations_with_random_anchors(id_to_sentence, n_anchors=3)
        # アンカーとなる定義文を各relから1つ選び、それと他の特徴文をペアにして生成する場合:
        # combs, id_to_sentence_filtered = pairwise_sentence_combinations_with_anchor_from_each_rel(id_to_sentence, rel_to_ids)
        # アンカーとなる定義文を、5文以上あるrelから1つ選び、それと他の特徴文をペアにして生成する場合:
        # combs, id_to_sentence_filtered = pairwise_sentence_combinations_with_anchor_from_big_rel(id_to_sentence, rel_to_ids)
        # return # アンカーありのペア作成関数の動作確認のため、ここで一旦停止。生成ループは後で実行。

        for comb in tqdm(combs, desc="Sentence Pairs", leave=False):
            # 🔸 もしこのペアについて既に生成済みならスキップ
            if identify_gen_dt is not None:
                id_i, id_j = comb
                already_done = False
                for record in identify_gen_dt:
                    if record["id_1"] == id_i and record["id_2"] == id_j:
                        already_done = True
                        break
                if already_done:
                    print(f"  【Skipped】 Pair ({id_i}, {id_j}) already generated.")
                    continue

            id_i, id_j = comb
            sent1 = id_to_sentence_filtered[id_i]
            sent2 = id_to_sentence_filtered[id_j]

            print(f"Pair: ({id_i}, {id_j})")
            print(f"  1: {sent1}")
            print(f"  2: {sent2}")


            # *** Geminiによる特徴ペア→元の固有名詞判定 の生成 ***
            # *** prompt構築 
            prompt = gen_guess_concept_prompt_base.replace("{sentence_1}", sent1)
            prompt = prompt.replace("{sentence_2}", sent2)
            # print(f"【prompt】\n{prompt}")
                
            # *** Geminiで生成 
            response = gen_with_google_genai_api(client,prompt, schema, temperature, topP, model, max_retries)

            if response is not None:
                record = {
                    "response_id": response.response_id,
                    "model_version": response.model_version,
                    "text": response.text,
                    "parsed": response.parsed,
                    "usage_metadata": {
                        "prompt_token_count": response.usage_metadata.prompt_token_count,
                        "candidates_token_count": response.usage_metadata.candidates_token_count,
                        "total_token_count": response.usage_metadata.total_token_count,
                    },
                }
                gen_succeeded = True
            else:
                record = {
                    "error": "Failed to generate a valid response after maximum retries.",
                    "prompt": prompt,
                    "schema": schema,
                    "model": model,
                    "max_retries": max_retries,
                }
                gen_succeeded = False

            # *** 生成文のまま保存 
            out_path = os.path.join(generated_savedir, f"{target_concept.replace(' ', '_')}.jsonl")
            handle_one_generation(id_i, id_j, sent1, sent2, record, Path(out_path))
            print(f"Pair ({id_i}, {id_j}) for '{target_concept}' saved to {out_path}")

            time.sleep(delay_between_requests)  # APIリクエスト間の遅延

            # break
        # break
    print("All done.")



def logic_ok(o: dict) -> bool:
    """固有名詞を特定できるならその固有名詞を(1word以上持つ), 特定できないならnullにするという条件付き制約を満たしているかチェックする.
    どちらかを満たしていればTrue, そうでなければFalseを返す.
    Args:
      o (dict): チェック対象のjsonオブジェクト
    Returns:
      bool: 条件付き制約を満たしていればTrue, そうでなければFalse
    例:
      >>> logic_ok({"identifiable": True, "proper_noun": "Eiffel Tower", "confidence": 0.95})
      True
      >>> logic_ok({"identifiable": False, "proper_noun": None, "confidence": 0.90})
      True
      >>> logic_ok({"identifiable": True, "proper_noun": "", "confidence": 0.80})
      False
    """
    # 型の最低保証（念のため）
    if not isinstance(o.get("identifiable"), bool):
        return False
    if not (o.get("proper_noun") is None or isinstance(o.get("proper_noun"), str)):
        return False
    conf = o.get("confidence")
    if not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
        return False

    # ここが本題：条件付き制約
    if o["identifiable"]:
        return isinstance(o["proper_noun"], str) and len(o["proper_noun"].strip()) > 0
    else:
        return o["proper_noun"] is None


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    1レコード=1行(JSON)で追記する。
    ensure_ascii=False で日本語をそのまま保存。
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def save_result(
    id_1: int,
    id_2: int,
    sentence_1: str,
    sentence_2: str,
    response: dict,
    # parsed: Optional[Dict[str, Any]],
    # ok: bool,
    path: Path = Path("results.jsonl"),
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "id_1": id_1,
        "id_2": id_2,
        "sentence_1": sentence_1,
        "sentence_2": sentence_2,
        # "ok": ok,                 # 後処理の整合性チェックに通ったか
        # "parsed": parsed,         # パースできたJSON（できなければNone）
        # "raw_text": raw_text,     # モデルの生出力（デバッグ用）
        "response_info": response, # モデルの応答全体（生出力+メタデータ）
    }
    if extra:
        record.update(extra)
    append_jsonl(path, record)


# --- 例：レスポンスからパースして保存する ---
def handle_one_generation(
        id_1: int, id_2: int,
        sentence_1: str, sentence_2: str, 
        response: dict, 
        path: Path = Path("results.jsonl")) -> None:
    

    # ok = logic_ok(parsed) if isinstance(parsed, dict) else False
    save_result(
        id_1, id_2, 
        sentence_1, sentence_2, 
        response, 
        # parsed, 
        # ok, 
        path
        )
    
# **** 検証ペア作成用関数 ****
# 全通りのペアを作成する場合(ペア数はかなり多くなるので注意):
def pairwise_sentences_combinations(id_to_sentence):

    id_to_sentence_filtered = {}
    for id, sentence in id_to_sentence.items():
        if 'unknown' not in sentence:
            id_to_sentence_filtered[id] = sentence
        # else:
        #     print(f"\tExcluded id {id} due to 'unknown' in sentence.")

    indices = sorted(list(id_to_sentence_filtered.keys()))
    num_sentences = len(indices)

    # 全通りのペアを作成. 35個から2個を選ぶ組み合わせ
    combs = list(itertools.combinations(indices, 2))

    # 結果の確認
    print(f"sentence num: {num_sentences}, valid comb num: {len(combs)}, e.g. combs: {combs[:10]}...")
    return combs, id_to_sentence_filtered



# アンカーとなる定義文を、ランダムにn個選び、それと他の特徴文をペアにして生成する場合:
def pairwise_sentence_combinations_with_random_anchors(id_to_sentence, n_anchors=3):
    # ** anchorは、ランダムにn_anchorsつ選ぶ.
    random.seed(RANDOM_SEED)  # シャッフルの再現性のためにシードをここで設定

    id_to_sentence_filtered = {id: sentence for id, sentence in id_to_sentence.items() if 'unknown' not in sentence.lower()}
    if len(id_to_sentence_filtered) < n_anchors:
        print(f"Warning: Not enough valid sentences to select {n_anchors} anchors. Reducing number of anchors to {len(id_to_sentence_filtered)}.")
        return None, id_to_sentence_filtered
    
    anchor_ids = random.sample(list(id_to_sentence_filtered.keys()), n_anchors)

    
    # ** 各anchor_idについて、他のidとペアを作る
    combs = []
    for anchor_id in anchor_ids:
        anchor_sentence = id_to_sentence_filtered[anchor_id]
        other_ids = [id for id in id_to_sentence_filtered.keys() if id != anchor_id]

        for other_id in other_ids:
            other_sentence = id_to_sentence_filtered.get(other_id)
            if not other_sentence:
                continue
            # ここで(anchor_sentence, other_sentence)のペアを生成して処理する
            combs.append((anchor_id, other_id))
            print(f"Anchor ID: {anchor_id}")
            print(f"  Anchor Sentence: {anchor_sentence}")
            print(f"  Other Sentence: {other_sentence}")

    print(f"Total pairs with anchors: {len(combs)}")
    return combs, id_to_sentence_filtered



# アンカーとなる定義文を各relから1つ選び、それと他の特徴文をペアにして生成する場合:
def pairwise_sentence_combinations_with_anchor_from_each_rel(id_to_sentence, rel_to_ids):
    # ** anchorは、各relから1つ選ぶ.
    anchor_ids = []
    id_to_sentence_filtered = {}
    for rel, ids in rel_to_ids.items():
        id_to_sentence_filtered_for_rel = {id: id_to_sentence[id] for id in ids if 'unknown' not in id_to_sentence[id].lower()}
        if id_to_sentence_filtered_for_rel == {}:
            print(f"\tAll sentences for relation '{rel}' contain 'unknown'. Skipping this relation.")
            continue
        id_to_sentence_filtered.update(id_to_sentence_filtered_for_rel)
        ids_filtered = list(id_to_sentence_filtered_for_rel.keys())

        # rel中の最初のidをアンカーにする
        anchor_id = ids_filtered[0]
        # anchor_sentence = id_to_sentence_filtered[anchor_id]
        anchor_ids.append(anchor_id)
    
    # ** 各anchor_idについて、他のidとペアを作る
    combs = []
    for anchor_id in anchor_ids:
        anchor_sentence = id_to_sentence_filtered[anchor_id]
        other_ids = [id for id in id_to_sentence_filtered.keys() if id != anchor_id]

        for other_id in other_ids:
            other_sentence = id_to_sentence_filtered.get(other_id)
            if not other_sentence:
                continue
            # ここで(anchor_sentence, other_sentence)のペアを生成して処理する
            combs.append((anchor_id, other_id))
            print(f"Anchor ID: {anchor_id}")
            print(f"  Anchor Sentence: {anchor_sentence}")
            print(f"  Other Sentence: {other_sentence}")

    print(f"Total pairs with anchors: {len(combs)}")
    return combs, id_to_sentence_filtered

# アンカーとなる定義文を、5文以上あるrelから1つ選び、それと他の特徴文をペアにして生成する場合:
def pairwise_sentence_combinations_with_anchor_from_big_rel(id_to_sentence, rel_to_ids):
    # ** anchorは、各relから1つ選ぶ.
    anchor_ids = []
    id_to_sentence_filtered = {}
    for rel, ids in rel_to_ids.items():
        id_to_sentence_filtered_for_rel = {id: id_to_sentence[id] for id in ids if 'unknown' not in id_to_sentence[id].lower()}
        if len(id_to_sentence_filtered_for_rel) < 5:
            print(f"\tRelation '{rel}' has fewer than 5 valid sentences. Skipping.")
            continue
    
        id_to_sentence_filtered.update(id_to_sentence_filtered_for_rel)
        ids_filtered = list(id_to_sentence_filtered_for_rel.keys())

        # rel中の最初のidをアンカーにする
        anchor_id = ids_filtered[0]
        # anchor_sentence = id_to_sentence_filtered[anchor_id]
        anchor_ids.append(anchor_id)
    
    # ** 各anchor_idについて、他のidとペアを作る
    combs = []
    for anchor_id in anchor_ids:
        anchor_sentence = id_to_sentence_filtered[anchor_id]
        other_ids = [id for id in id_to_sentence_filtered.keys() if id != anchor_id]

        for other_id in other_ids:
            other_sentence = id_to_sentence_filtered.get(other_id)
            if not other_sentence:
                continue
            # ここで(anchor_sentence, other_sentence)のペアを生成して処理する
            combs.append((anchor_id, other_id))
            print(f"Anchor ID: {anchor_id}")
            print(f"  Anchor Sentence: {anchor_sentence}")
            print(f"  Other Sentence: {other_sentence}")

    print(f"Total pairs with anchors: {len(combs)}")
    return combs, id_to_sentence_filtered


if __name__ == "__main__":
    print("=== Generating guessed proper nouns from sentence pairs ===")
    main()