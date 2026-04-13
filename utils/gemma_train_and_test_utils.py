
import random
import os
import sys

# ===== Third-party =====
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn.functional as F

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