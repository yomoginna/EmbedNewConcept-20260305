

import os
import sys
import re

# ===== Third-party =====
import numpy as np
import torch

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
sys.path.append(project_root)

from utils.wikipedia_api_utils import load_wiki_text
from utils.handle_text_utils import split_text_into_sentences

# wiki_page_save_dir = os.path.join(project_root, 'data', 'wiki_pages')
# dont_get_new_wiki_flag = False # False #True # もう新しいwikiページを読み込みたくない場合はTrue. すでに保存済みのwikiページがあるpropernounのみにフィルタリングする.





def find_subsequence(sequence, subsequence):
    """
    sequence 内で subsequence が始まる index を返す
    見つからなければ -1
    """
    n = len(subsequence)

    for i in range(len(sequence) - n + 1):
        if sequence[i:i+n] == subsequence:
            return i
    print(f"Error: Subsequence ***{subsequence}*** not found in sequence ***{sequence}***.")
    return -1



def get_span_subseq_in_fullseq_use_offset(
    text,
    target_text,
    tokenizer,
):
    """
    一番最後に見つけたtarget_textのspanを、tokenizerのoffsetを使ってfull sequence内で特定する
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,    # offset: tokenizerが作った各tokenが、元の文字列の何文字目から何文字目に対応しているかを表す位置情報
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]

    # * text内で最後に出現するtarget_textのspan(何文字目から何文字目まで)を特定する *
    char_start = text.rfind(target_text)    # textの後からtarget_textを探す. text内で最後に出現するtarget_textのspanを特定のため.
    if char_start == -1:
        raise ValueError(f"target_text not found in text: {target_text}")
    char_end = char_start + len(target_text)

    # * text内で最後に出現するtarget_textのspanとoverlapするtokenをoffsetを使って特定する *
    token_indices = []
    for token_idx, (start, end) in enumerate(offsets):
        # 「token の範囲 (start, end) と target の範囲 (char_start, char_end) が重なっているか」を判定
        if not (end <= char_start or start >= char_end):
            token_indices.append(token_idx)

    if len(token_indices) == 0:
        raise ValueError(f"No tokens matched target_text: {target_text}")

    # start_token_idx = token_indices[0]
    # end_token_idx = token_indices[-1] + 1  # endはexclusiveにするために+1
    start_token_idx = min(token_indices)
    end_token_idx = max(token_indices) + 1  # endはexclusiveにするために+1

    return start_token_idx, end_token_idx



# ** こちらだと、full_idsとsubseq_idsが完全に一致する必要があるため、tokenizerの違いなどでうまくいかない可能性がある。その場合は上のget_span_subseq_in_fullseq_use_offsetを使う
def get_span_subseq_in_fullseq(
    full_ids,
    target_text,
    tokenizer,
):
    """
    Parameters
    ----------
    full_ids : list
        文全体の token ids
    target_text : str
        mean pooling したい単語列
    tokenizer :
        HuggingFace tokenizer

    Returns
    -------
    pooled_vec : torch.Tensor
        shape: (hidden_dim,)
    """

    # # 文全体 token ids
    # full_ids = tokenizer.encode(text, add_special_tokens=False)

    # 対象 span token ids
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    # subsequence 検索
    start_idx = find_subsequence(full_ids, target_ids)

    if start_idx == -1:
        print(f"decoded full_ids: ***{tokenizer.decode(full_ids)}***")
        print(f"decoded subsequence: ***{tokenizer.decode(target_ids)}***")
        raise ValueError(f"target_text not found: {target_text}")

    end_idx = start_idx + len(target_ids)
    return start_idx, end_idx




# =====================================================
# pool_hs_type に応じた hidden state を取りvecを作成する関数
# =====================================================
@torch.no_grad()
def extract_hidden_states(
    model, 
    tokenizer, 
    text_list, 
    pool_hs_type,
    batch_size=8, 
    layer_index=None, 
    mean_pool_target_texts=None, 
    print_flag=False):
    """
    pool_hs_typeに応じて hidden state を返す。

    Args:
        model: 言語モデル
        tokenizer: トークナイザ
        text_list: テキストのリスト
        pool_hs_type: hidden stateのpooling方法 ("eos", "last_token, "mean_pool", "dot" ) + ("repeat_mean_pool" "target_seq_repeat_mean_pool" も追加) 2026/05/22
        batch_size: バッチサイズ
        mean_pool_target_texts: mean_poolの対象となるテキストのリスト. pool_hs_typeが"target_seq_repeat_mean_pool"のときのみ使用
        layer_index: int or list or 'all'. 隠れ状態を抽出する層のインデックス. -1: 最終層
        print_flag: 抽出するhidden stateの位置を確認するためのprint文を表示するかどうか

    Returns:
        np.ndarray of shape (N, hidden_dim)
    """
    if layer_index is None:
        raise ValueError("layer_index must be specified")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


    all_vecs = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i+batch_size]
        if mean_pool_target_texts is not None:
            batch_mean_pool_target_texts = mean_pool_target_texts[i:i+batch_size]
        

        # ** 前処理 **
        if pool_hs_type == "eos":
            """eosの隠れ状態を取得する方法"""
            # EOS を明示的に末尾へ追加
            batch_texts = [text + tokenizer.eos_token for text in batch_texts]

        # ** tokenize and generate **
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False #last_token_is_eos#LAST_TOKEN_IS_EOS, -> pool_hs_type == "eos"の場合は明示的にeosを追加済みなので、ここをTrueにするとeosが重複して2つ付く可能性がある。そのためここはFalseで良い。
        ).to(model.device) 

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states は tuple:
            # 0: embedding出力, 1..L: 各層出力
            hs = outputs.hidden_states

        # *** pool_hs_type に応じて、vectorを抽出 ***
        if pool_hs_type == "eos":
            # 各系列について EOS token の最後の出現位置を取る   
            eos_mask = (input_ids == tokenizer.eos_token_id)


        for s_id in range(input_ids.size(0)):
            # 1 が立っている位置を取得 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] -> valid_pos = [6, 7, 8, 9, 10, 11] 
            valid_pos = torch.nonzero(attention_mask[s_id], as_tuple=False).squeeze(-1)
            # print(f"valid_pos for batch {s_id}: {valid_pos}")
            if valid_pos.numel() == 0:
                # 全部 padding の場合
                pos_begin = 0
                pos_end = 0
            else:
                pos_begin = valid_pos[0].item()
                pos_end = valid_pos[-1].item() + 1   # slice用に end は exclusive


            if pool_hs_type == "eos":
                eos_positions = torch.where(eos_mask[s_id])[0]
                if len(eos_positions) == 0:
                    raise ValueError(f"EOS token が見つかりません: {batch_texts[s_id]}")
                eos_pos = eos_positions[-1].item()
                pos_begin = eos_pos
                pos_end = eos_pos + 1

            elif pool_hs_type == "mean_pool":
                """text全体の隠れ状態を平均する方法"""
                # pos_begin, pos_end はすでに valid part の範囲を示しているので、その範囲内で平均する
                pass

            elif pool_hs_type == "target_seq_repeat_mean_pool":
                """textを2回repeatし、2文目のtextにおける、target_text(concept名等)部分のみの隠れ状態を平均する方法。target_textが2tokens以上の場合は、そのtokensで平均する"""
                if batch_mean_pool_target_texts is None:
                    raise ValueError("mean_pool_target_texts must be provided when pool_hs_type is 'target_seq_repeat_mean_pool'")
                text = batch_texts[s_id]
                target_text = batch_mean_pool_target_texts[s_id]  # target_textsはtext_listと同順でtarget_textが入っているリストであることを想定
                pos_begin, pos_end = get_span_subseq_in_fullseq_use_offset(text, target_text, tokenizer)

            elif pool_hs_type == "last_token":
                """最後のtokenの隠れ状態を取得する方法"""
                pos_begin = pos_end - 1

            elif pool_hs_type == "repeat_mean_pool":
                    # 同じ文を2回繰り返してプロンプトとする場合は、2回目の文のみの隠れ状態を平均する
                    # seq_first_half_len = seq_len // 2   # 1文が5tokens → id:0,1,2,3,4 が1文目、id:5,6,7,8,9 が2文目の場合、seq_len=10, seq_first_half_len=5 となる
                    pos_begin_second_sent = (pos_begin + pos_end) // 2 # == pos_begin + (pos_end - pos_begin) / 2
                    pos_begin = pos_begin_second_sent

            elif pool_hs_type == "dot":
                # ** dot (.) 位置のベクトルを抽出 (テンソルの中で「0でない（≒True）」要素のインデックスを取得する, その際tupleではなくテンソルで返すためにas_tuple=Falseを指定, さらに次元が1のときに余分な次元を削除するためにsqueeze(-1)を指定)
                # dot_pos = (input_ids[s_id] == tokenizer.convert_tokens_to_ids('.')).nonzero(as_tuple=False).squeeze(-1)
                dot_ids = tokenizer.encode(".", add_special_tokens=False)
                if len(dot_ids) != 1: raise ValueError(f"'.' is not a single token: {dot_ids}")
                dot_pos = (input_ids[s_id] == dot_ids[0]).nonzero(as_tuple=False).squeeze(-1)

                if dot_pos.numel() == 0:
                    # dot がない場合はerror
                    err_text = f"Batch {s_id} のテキスト '{batch_texts[s_id]}' にはdot(.)が含まれていません。dot(.)を含むテキストを使用してください。"
                    raise ValueError(err_text)
                pos_begin = dot_pos[-1].item()  # 最後のdotの位置を使用
                pos_end = pos_begin + 1

            else:
                raise ValueError(f"Unknown pool_hs_type: {pool_hs_type}")
            

            if print_flag:
                # どの位置のtokenの隠れ状態が使われるのかを確認するためのprint文
                print(f"Batch {s_id} text: '{batch_texts[s_id]}'")
                print(f"Token IDs: {input_ids[s_id].tolist()}")
                print(f"pos_begin: {pos_begin}, pos_end: {pos_end}")
                print(f"\tattention_mask: {attention_mask[s_id]},\n\t valid_pos: {valid_pos}, \n\t valid part in text: {input_ids[s_id][pos_begin:pos_end]}")

            # ** 結果のvecをall_vecsに追加
            if type(layer_index) == int:
                # layer_indexが1つだけ指定された場合
                layer_hs = hs[layer_index]      # (B, T, H)
                vec = layer_hs[s_id, pos_begin:pos_end, :].mean(dim=0)  # (H,)
                all_vecs.append(vec.detach().float().cpu().numpy()) # all_vecs: (T, D)  Tはテキスト数, Dは隠れ状態の次元
            else:
                if layer_index == 'all':
                    # layer_indexが'all'の場合は、全ての層の隠れ状態を抽出して連結する
                    layer_indices = list(range(len(hs)))
                else:
                    # layer_indexが複数、intで指定された場合は、指定された全ての層の隠れ状態を抽出して連結する
                    layer_indices = layer_index

                # layer_indexが複数指定された場合は、指定された全ての層の隠れ状態を抽出して連結する
                layer_to_vecs = []
                for l_idx in layer_indices:
                    layer_hs = hs[l_idx]      # (B, T, H)
                    vec = layer_hs[s_id, pos_begin:pos_end, :].mean(dim=0)  # (H,)
                    layer_to_vecs.append(vec.detach().float().cpu().numpy())
                all_vecs.append(layer_to_vecs)  # (T, H, D) Tはテキスト数, Hは層の数, Dは隠れ状態の次元

    return np.stack(all_vecs, axis=0)



def get_concept_containing_text_using_wiki_summary(concept, target_name_to_replace_with_concept=None):
    """conceptが含まれるwikisummary内の文、もしくは concept名を target_name_to_replace_with_concept に置換した文を取得する関数. 
    大抵の wiki page は、概念名とその説明を含む1文から始まっているため、冒頭の文に概念名が含まれればその文を抽出する。含まれなければ、" is the " を含む文を探し、その前の部分を概念名に置換する。
    例:
    - conceptが"Apple Inc."であれば、Wikipediaのsummaryの中から"Apple Inc."を含む文を抽出し、その中で「最初の文」を返す. 
    - もし"Apple Inc."を含む文が見つからない場合は、" is the "を含む文を探し、その前の部分を"Apple Inc."に置換して返す. それも見つからない場合は、summary全体を返す.

    Args:
        concept (str): 埋め込みたい概念名
        target_name_to_replace_with_concept (str, optional): concept名に置換する対象名. 例えば，concept="banana", wiki first sent = "banana is yellow" の場合，この最初のbananaを何に置き換えるか．noneの場合は置換なし，none以外の場合は置換する．

    Returns:
        str: conceptを埋め込む文
    """
    special_case_dic = {
        "&": "and", # wikiタイトルでは 'Adam & Eve' だが、wiki page内文章では 'Adam and Eve' と表現されていた
        "and": "&",

    }
    

    # 辞書にまだ保存されていなければ、data dir もしくは wiki apiから取得して、self.propnoun_to_wikisummaryに格納する
    summary = load_wiki_text(concept, text_type="summary")
    if not summary:
        return None

    # ** concept名が含まれる文だけを抽出し，さらにその中の一番最初の文を収集する．大文字小文字の違いは無視し，記号([]',.')なども無視して比較する．例えば、conceptが"Apple Inc."であれば、summaryの中から"Apple Inc."を含む文を抽出し、その中で最初の文を収集する．このとき、"apple inc"や"Apple, Inc."などもconcept名とみなす．
    base_concept_pattern = re.escape(concept)                            # 概念名が含まれるかどうかを調べるパターン。concept名を正規表現の特殊文字をエスケープしてパターン化
    brackets_dup_pattern = re.compile(r'\s*\(.*?\[.*?\].*?\)\s*')   # ( ... [..] ... ) のような()内に[]が入るパターン. wiki title(data名) と wiki page文章内の綴りが異なっても、( [])内に読み方が含まれれば、その前部分が概念名であるとわかる。例: concept名: 'House of Hohenstaufen' wiki文章: "The Hohenstaufen dynasty (, US also , German: [ˌhoːənˈʃtaʊfn̩]), also known as the Staufer,  ..."
    is_the_pattern = re.compile(r'(.+?)\s+is\s+the\s+', re.IGNORECASE)      # " is the " が含まれるかどうかを調べるパターン
    brackets_colon_pattern = re.compile(r"\([^()]*[:;][^()]*\)") # (en: Apple Inc.) のような、()内に:が入るパターン. "\([^()]*[:;][^()]*\)"
    brackets_stylised_pattern = re.compile(r'\s*\(.*?stylised.*?\)\s*') # (stylised as ABC) のようなパターン. これも概念名の一部ではないので削除する.


    # concept pattern に含まれる一部の文字が、wiki page内では他の文字で表されている場合があるため、その場合のpatternも作成しておく。
    concept_patterns = [base_concept_pattern]
    for k, v in special_case_dic.items():
        if k in concept:
            concept_patterns.append(re.escape(concept.replace(k, v)))


    sentences = split_text_into_sentences(summary)  # 文の区切りでsummaryを分割
    first_concept_sentence = None

    ## ① ベースとして、概念名orその置換パターンを含む文を探す
    concept_sentences = [
        s for s in sentences
        if any(re.search(p, s, re.IGNORECASE) for p in concept_patterns)    # 概念名orその置換パターンの何かしらに引っ掛かればOK
    ]

    if concept_sentences:
        first_concept_sentence = concept_sentences[0]
        # もし, 基本の概念名ではなく辞書を元に置換した概念名でマッチした場合は、それを基本の概念名に戻す。
        for concept_pattern in concept_patterns:
            if re.search(concept_pattern, first_concept_sentence, re.IGNORECASE):
                first_concept_sentence = re.sub(concept_pattern, concept, first_concept_sentence, flags=re.IGNORECASE)

    # ② 概念名を含む文が見つからない場合 → ( []) を含む文を探す。概念名が違う言語で書かれている場合などがある。この場合、他言語の概念名 (名前 [発音]) の表記であることがあるため、そのパターンを探し、冒頭から()までを概念名に置換する
    if first_concept_sentence is None:
        brackets_dup_sentences = [s for s in sentences if brackets_dup_pattern.search(s)]
        if brackets_dup_sentences:
            first_brackets_dup_sentence = brackets_dup_sentences[0]
            # その文の中の、brackets_dup_patternにマッチする部分とそれ以前を概念名に置換する
            modified_sentence = re.sub(rf"^.*?{brackets_dup_pattern.pattern}", concept, first_brackets_dup_sentence)           
            first_concept_sentence = modified_sentence
            print(f"\t No sentence containing concept '{concept}' was found. Using modified '( ... [..] ... )' sentence: \n\t\t{first_concept_sentence}\n")

    # ③ 概念名を含む文が見つからない場合 → " is the "を含む文を探して概念名を含む文を作る．
    # 例えば 'Arnhem Bridge' のsummaryの1文目は "John Frost Bridge (John Frostbrug in Dutch) is the road bridge over the Lower Rhine at Arnhem, in the Netherlands." である．この is the の前の部分を，concept名に置換して使う．
    if first_concept_sentence is None:
        # print(f"No sentence containing concept '{concept}' was found in the summary. Trying to find a sentence containing 'is the' to use as a template for the concept name.\n")
        is_the_sentences = [s for s in sentences if is_the_pattern.search(s)]
        if is_the_sentences:
            first_is_the_sentence = is_the_sentences[0]
            # first_is_the_sentenceのうち，「is the とマッチした部分」を，「concept + " is the "」に置換する
            modified_sentence = is_the_pattern.sub(concept + " is the ", first_is_the_sentence)
            first_concept_sentence = modified_sentence
            print(f"\t No sentence containing concept '{concept}' was found. Using modified 'is the' sentence: \n\t\t{first_concept_sentence}\n")

    if first_concept_sentence is None:   # 以上で解決できなければ、raise error. or Noneを返す
        # raise ValueError(f"No sentence containing concept '{concept}' was found in the summary, and no sentence containing 'is the' was found either. Please check the summary for concept '{concept}': \n {summary}")
        return None

    # [memo] この時点で必ず，concept名 を含む文が1つ，first_concept_sentenceとして格納されていることになる．

    # ** 置換先の名前が登録されている場合は、first_concept_sentenceのうち、concept_patternにマッチした部分をtarget_name_to_replace_with_conceptに置換する処理
    if target_name_to_replace_with_concept is not None:
        first_concept_sentence = re.sub(
                    re.escape(concept),
                    target_name_to_replace_with_concept,
                    first_concept_sentence,
                    flags=re.IGNORECASE,
                )
    
    # ** 文をcleaning
    first_concept_sentence = re.sub(brackets_dup_pattern, " ", first_concept_sentence)      # ( ... [..] ... ) パターンをスペースに置換
    first_concept_sentence = re.sub(brackets_colon_pattern, " ", first_concept_sentence)    # ( ... : ... ) パターンをスペースに置換
    first_concept_sentence = re.sub(brackets_stylised_pattern, " ", first_concept_sentence) # (stylised ...) パターンをスペースに置換
    first_concept_sentence = re.sub(r'\(\)', ' ', first_concept_sentence)                   # ()のみのパターンをスペースに置換
    first_concept_sentence = re.sub(r" {2,}", " ", first_concept_sentence)                  # 連続するスペースを1つのスペースに置換
    first_concept_sentence = re.sub(r"\s+([,.?!])", r"\1", first_concept_sentence)          # スペース + 句読点のパターンを、スペースなしの句読点に置換 (例. "The Ayyubid dynasty , also known as ..." -> "The Ayyubid dynasty, also known as ...")
    first_concept_sentence = first_concept_sentence.strip()                                 # 先頭と末尾
        
    return first_concept_sentence
    

def load_mem_vec(model, mem_save_path, memTokenIds):
    vecs = np.load(mem_save_path)

    embedding_layer = model.get_input_embeddings()
    weight = embedding_layer.weight

    vecs_tensor = torch.as_tensor(
        vecs,
        dtype=weight.dtype,
        device=weight.device,
    )

    if vecs_tensor.shape[0] != len(memTokenIds):
        raise ValueError(
            f"Number of vectors ({vecs_tensor.shape[0]}) does not match "
            f"number of token IDs ({len(memTokenIds)})."
        )

    # 特定の token ID の位置にベクトルを上書き
    with torch.no_grad():
        weight[memTokenIds] = vecs_tensor

    print(f"Loaded trained vectors from {mem_save_path} into model embedding layer.")


          
# def insert_mem_vec_into_model(model, mem_vecs, memTokenIds, ):
