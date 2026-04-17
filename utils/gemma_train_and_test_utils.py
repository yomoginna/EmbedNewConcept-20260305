
import random
import os
import sys
# import math

# ===== Third-party =====
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
# from tqdm import tqdm
import torch
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import wandb

# プロジェクトのutils追加
project_root = os.path.join(os.path.dirname(__file__), "..") # os.path.dirname(__file__): スクリプト自身のパス
sys.path.append(project_root)




def fix_seed(seed=0):
    """Fix random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_mem_vec(model, memTokenIds, mem_save_path):
    os.makedirs(os.path.dirname(mem_save_path), exist_ok=True)

    # memTokenIdsの指定した要素の値(=weightのindex)の並び順のまま取り出され, 勝手にソートされることは無い
    try:
        vecs = (
            model.model.embed_tokens.weight[memTokenIds]
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        )
    except AttributeError:
        vecs = (
            model.model.language_model.embed_tokens.weight[memTokenIds]
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
        )
    np.save(mem_save_path, vecs)



def load_mem_vec(model, mem_save_path, memTokenIds):
    vecs = np.load(mem_save_path)
    try:
        vecs_tensor = torch.tensor(vecs, dtype=model.model.embed_tokens.weight.dtype, device=model.model.embed_tokens.weight.device)
    except:
        vecs_tensor = torch.tensor(vecs, dtype=model.model.language_model.embed_tokens.weight.dtype, device=model.model.language_model.embed_tokens.weight.device)
    # print(f"Vector shape: {vecs_tensor.shape}, Embedding weight shape: {model.model.embed_tokens.weight.shape}")

    # 特定の token ID の位置にベクトルを上書き
    with torch.no_grad():
        for i, token_id in enumerate(memTokenIds):
            try:
                model.model.embed_tokens.weight[token_id] = vecs_tensor[i]
            except:
                model.model.language_model.embed_tokens.weight[token_id] = vecs_tensor[i]

    print(f"Loaded trained vectors from {mem_save_path} into model embedding layer.")
          


def evaluateModel(model, tokenizer, evalInputs, evalOutputTexts, verbose=False):
    """
    Evaluate the model on the given inputs and outputs.
    """
    model.eval()
    # total_val_loss = 0.0
    # total_val_tokens = 0

    maxNewTokens = 10
    numCorrect = 0
    print('Evaluating on %d samples...'%len(evalInputs))
    with torch.no_grad():
        for i in range(len(evalInputs)):
            generation = model.generate(
                                    torch.LongTensor([evalInputs[i]]).to(model.device),
                                    max_new_tokens=maxNewTokens, 
                                    do_sample=False,
                                    repetition_penalty=1.05,
                                )[0]
            decodedGeneration = tokenizer.decode(generation)


            if verbose and i//2==0 or i==len(evalInputs)-1: # if verbose and i==0 or i==len(evalInputs)//2 or i==len(evalInputs)-1:
                print('--- Sample %d ---'%i)
                # print('\tP:', decodedGeneration)
                # print('\tT:', evalOutputTexts[i])
                print(decodedGeneration.startswith(evalOutputTexts[i]))
            
            if decodedGeneration.startswith(evalOutputTexts[i]):
                # もし生成テキストが正解テキストと一致したら
                numCorrect += 1

    acc = numCorrect / len(evalInputs)
    return acc



def extract_probability_of_option_numbers(target_logits, tokenizer, num_options):
    """選択肢の数字の生成確率を抽出する
    Args:
        target_logits (torch.Tensor): モデルの出力ロジット
        tokenizer (transformers.AutoTokenizer): トークナイザ
        num_options (int): 選択肢の数
    Returns:
        log_prob_dicts (List[Dict[str, float]]): 各選択肢番号に対する生成log確率辞書のリスト. e.g. [{"1": -1.2, "2": -0.5}, {"1": -0.3, "2": -1.5}, ...]
        prob_dicts (List[Dict[str, float]]): 各選択肢番号の生成確率辞書のリスト
    """
    log_probs = F.log_softmax(target_logits, dim=-1) # logなので, 確率0~1.0は, マイナスor0 になる. 確率が小さいほどlogも小さくなる.
    probs = torch.exp(log_probs)  # log_probsを確率に変換
    # print(f"log_probs shape: {log_probs.shape}")  # (batch_size, vocab_size)

    # 選択肢のトークンIDを取得
    number_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, num_options + 1)]
    # 各選択肢番号のトークンIDに対応するlog確率を取得
    token_log_probs_of_all_q = log_probs[:, number_token_ids]  # 各テスト問題に対する、各選択肢番号tokenのlog確率を取得
    token_prob_of_all_q = probs[:, number_token_ids]  # 各テスト問題に対する、各選択肢番号tokenの確率を取得
    # print(f"token_scores_lst shape: {token_scores_lst.shape}")  # (batch_size, num_options)

    # 各選択肢番号のlog確率を取得
    log_prob_dicts = []
    for token_log_probs_of_q in token_log_probs_of_all_q:
        num_2_log_prob = {}
        for i, score in enumerate(token_log_probs_of_q): # for num_token_id, score in zip(number_token_ids, token_log_probs_of_q):
            num_2_log_prob[f"{i + 1}"] = score.item() # num_2_log_prob[f"{tokenizer.convert_ids_to_tokens(num_token_id)}"] = score.item()
        log_prob_dicts.append(num_2_log_prob)
    
    prob_dicts = []
    for q_token_probs_of_q in token_prob_of_all_q:
        num_2_prob = {}
        for i, prob in enumerate(q_token_probs_of_q):
            num_2_prob[f"{i + 1}"] = prob.item()
        prob_dicts.append(num_2_prob)

    return log_prob_dicts, prob_dicts



# ********* calculate metrics **********
def calculate_metrics(y_pred_lst, y_true_lst):
    """各種scoreを計算する
    Args:
        y_pred_lst (list): モデルの予測結果リスト
        y_true_lst (list): 正解ラベルリスト
    Returns:
        dict: accuracy, precision, recall, F1スコア
    """
    # accuracy, precision, recall, F1の計算
    accuracy = accuracy_score(y_true_lst, y_pred_lst)
    precision = precision_score(y_true_lst, y_pred_lst, average='weighted', zero_division=0)
    recall = recall_score(y_true_lst, y_pred_lst, average='weighted', zero_division=0)
    f1 = f1_score(y_true_lst, y_pred_lst, average='weighted', zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1
    }




# =========================
# pool_hs_type に応じた hidden state を取りvecを作成する関数
# =========================
@torch.no_grad()
def extract_hidden_states(model, tokenizer, text_list, pool_hs_type, data_type, batch_size=8, layer_index=-1, print_flag=False):
    """
    各テキストの末尾にEOSを明示的に追加し、
    EOSトークン位置の hidden state を返す。

    Returns:
        np.ndarray of shape (N, hidden_dim)
    """
    all_vecs = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]

        if pool_hs_type == "eos":
            # EOS を明示的に末尾へ追加
            batch_texts = [text + tokenizer.eos_token for text in batch_texts]
        
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
            layer_hs = hs[layer_index]      # (B, T, H)
        

        # *** pool_hs_type に応じて、vectorを抽出 ***
        if pool_hs_type == "eos":
            # 各系列について EOS token の最後の出現位置を取る
            eos_mask = (input_ids == tokenizer.eos_token_id)

        for t_idx in range(input_ids.size(0)):

            # 1 が立っている位置を取得 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] -> valid_pos = [6, 7, 8, 9, 10, 11] 
            valid_pos = torch.nonzero(attention_mask[t_idx], as_tuple=False).squeeze(-1)
            # print(f"valid_pos for batch {t_idx}: {valid_pos}")
            if valid_pos.numel() == 0:
                # 全部 padding の場合
                pos_begin = 0
                pos_end = 0
            else:
                pos_begin = valid_pos[0].item()
                pos_end = valid_pos[-1].item() + 1   # slice用に end は exclusive


            if pool_hs_type == "eos":
                eos_positions = torch.where(eos_mask[t_idx])[0]
                if len(eos_positions) == 0:
                    raise ValueError(f"EOS token が見つかりません: {batch_texts[t_idx]}")
                eos_pos = eos_positions[-1].item()
                pos_begin = eos_pos
                pos_end = eos_pos + 1

            elif pool_hs_type == "last_token":
                pos_begin = pos_end - 1


            elif pool_hs_type == "mean_pool":
                if data_type == "wiki_summary_repeat":
                    # wiki summaryを繰り返してプロンプトとする場合は、2回目の文のみの隠れ状態を平均する
                    # seq_first_half_len = seq_len // 2   # 1文が5tokens → id:0,1,2,3,4 が1文目、id:5,6,7,8,9 が2文目の場合、seq_len=10, seq_first_half_len=5 となる
                    pos_begin_second_sent = (pos_begin + pos_end) // 2 # == pos_begin + (pos_end - pos_begin) / 2
                    pos_begin = pos_begin_second_sent
                
            else:
                raise ValueError(f"Unknown pool_hs_type: {pool_hs_type}")
            
            vec = layer_hs[t_idx, pos_begin:pos_end, :].mean(dim=0)  # (H,)
            all_vecs.append(vec.detach().float().cpu().numpy())

            if print_flag:
                # どの位置のtokenの隠れ状態が使われるのかを確認するためのprint文
                print(f"pos_begin: {pos_begin}, pos_end: {pos_end}")
                print(f"\tattention_mask: {attention_mask[t_idx]},\n\t valid_pos: {valid_pos}, \n\t valid part in batch_text: {input_ids[t_idx][pos_begin:pos_end]}")


    return np.stack(all_vecs, axis=0)






# ******************************************* train functions ********************************************

def constructTrainSamples(concept_to_train_data_source, train_sample_format, conceptForFict2token_map, n_feat_in_a_sample=3, print_flag=False):
    train_samples = []
    for target_concept, train_data_list in concept_to_train_data_source.items():

        # 対応する空token名を取得. <unused0>など
        unused_token = conceptForFict2token_map[target_concept]
        
        for train_data in train_data_list:
            wiki_text_with_token = train_data['wiki_text_with_token']
            facts_with_token = train_data['facts_with_token']

            # factsの順番をランダムに入れ替える
            facts_with_token = random.sample(facts_with_token, len(facts_with_token))

            # *** featuresをn個ずつに分割して、1sampleあたり、summary1つ+特徴文n個の形式にする。最後の余りはそのまま1sampleにする。***
            train_sample_template = train_sample_format['train_sample']
            train_fact_sentence_template = train_sample_format['train_fact_sentence']

            for i in range(0, len(facts_with_token), n_feat_in_a_sample):
                fact_sentences = facts_with_token[i:i+n_feat_in_a_sample]
                fact_sentences_str = "\n".join([train_fact_sentence_template.format(fact_sentence=fs) for fs in fact_sentences])
                train_sample = train_sample_template.format(
                    unused_token=unused_token,
                    summary=wiki_text_with_token, 
                    fact_sentences=fact_sentences_str
                )
                train_samples.append(train_sample)
                if print_flag:
                    print(train_sample)
                    print("**********")
            
        # 各(架空)概念毎に1sample表示する
        print(f"Example train 1 sample for concept '{target_concept}':")
        print(train_samples[-1])
        print("**********")

    print(f"Total {len(train_samples)} train samples created.")
    return train_samples


def encodeTrainSamplesWithTokenizer(train_samples, tokenizer, padTokenId, device):
    maxLength = 0
    trainingData = []
    evalInputs = []
    evalOutputTexts = []
    temp_i = 0
    for sample in train_samples:
        # tokenized = tokenizer(sample)
        inputIds = tokenizer.encode(sample, add_special_tokens=True)

        if maxLength < len(inputIds):
            maxLength = len(inputIds)

        trainingData.append(inputIds) # 次token予測タスクのデータ
        evalInputs.append(inputIds[:-2]) # rel毎にルールベースで文にしているため，relの後の単語も同一になる文が多い. そのため, 最後の2tokenのみ(<word> + '.')の予測で可否を評価することにする
        evalOutputTexts.append(sample)

        if temp_i < 5:
            print(f"inputIds: {inputIds}")
            print(f"decoded inputIds: {tokenizer.decode(inputIds)}")
            print(f"evalInputIds: {evalInputs[-1]}") # 最後に追加したevalInputIdsを表示
            print(f"decoded evalInputIds: {tokenizer.decode(evalInputs[-1])}")
            print()
            temp_i += 1

    # ** padding ** 
    # : [[132, 45, 67], [23, 78]] -> [[132, 45, 67, [PAD], [PAD]], [23, 78, [PAD], [PAD], [PAD]]]
    trainingData = [t_ids + [padTokenId] * (maxLength - len(t_ids)) for t_ids in trainingData]
    trainingData = torch.LongTensor(trainingData).to(device)
    
    indices = list(range(len(trainingData)))
    return trainingData, evalInputs, evalOutputTexts, indices

