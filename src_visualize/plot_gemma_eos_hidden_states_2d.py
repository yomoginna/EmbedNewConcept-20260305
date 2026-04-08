
"""このスクリプトは、Gemmaモデルの特定の層におけるEOSトークン位置の隠れ状態を抽出し、PCAで次元削減してプロットするものです。

- 固有名詞リスト(proper_nouns)の各テキストの末尾にEOSトークンを追加して、その位置の隠れ状態を抽出します。(LAST_TOKEN_IS_EOS=Trueの場合)
- 抽出した隠れ状態をPCAで2次元に削減し、プロットします。
- プロットはoutput/hidden_state_pca_plotsディクトリに保存されます。

使用方法:
1. モデルのバージョンとサイズを設定します。
2. 固有名詞リスト(proper_nouns)を必要に応じて変更します。
3. スクリプトを実行して、プロットを生成します。

実行例:
uv run python3 src_visualize/plot_gemma_eos_hidden_states_2d.py

"""

import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

project_root = os.path.join(os.path.dirname(__file__), "..") #


# =========================
# 設定
# =========================


model_version = "3"
model_size = "4"
LAST_TOKEN_IS_EOS = True  # 固有名詞の最後にEOSトークンを追加して、その位置のhidden stateを取るかどうか。Falseの場合は最終トークン位置の隠れ状態を取得する
CUDA_VISIBLE_DEVICES = "2"
output_dir = os.path.join(project_root, "src_visualize", "output", "hidden_state_pca_plots")


MODEL_NAME = model_name = f"google/gemma-{model_version}-{model_size}b-it"

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 可視化したい固有名詞リスト
category_to_proper_nouns = {
    "都市": ["東京", "大阪", "京都", "渋谷"],
    "企業": ["ソニー", "トヨタ", "任天堂", "OpenAI"],
    "地理": ["富士山", "北海道"],
}

# どの層の hidden state を使うか
# -1 なら最終層
LAYER_INDEX = 9

# バッチサイズ
BATCH_SIZE = 8


# =========================
# フラット化
# =========================
proper_nouns = []
categories = []

for category, nouns in category_to_proper_nouns.items():
    for noun in nouns:
        proper_nouns.append(noun)
        categories.append(category)



# =========================
# モデル読み込み
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    # torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=None
).to(DEVICE)
model.eval()

# Gemma系で pad token が未設定な場合の保険
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.eos_token_id is None:
    raise ValueError("tokenizer.eos_token_id が見つかりません。")


# =========================
# EOS位置の hidden state を取る関数
# =========================
@torch.no_grad()
def extract_eos_hidden_states(text_list, batch_size=8, layer_index=-1):
    """
    各テキストの末尾にEOSを明示的に追加し、
    EOSトークン位置の hidden state を返す。

    Returns:
        np.ndarray of shape (N, hidden_dim)
    """
    all_vecs = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]

        # EOS を明示的に末尾へ追加
        batch_texts_with_eos = [text + tokenizer.eos_token for text in batch_texts]

        inputs = tokenizer(
            batch_texts_with_eos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=LAST_TOKEN_IS_EOS,
        ).to(model.device) 

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states は tuple:
            # 0: embedding出力, 1..L: 各層出力
            hs = outputs.hidden_states
            layer_hs = hs[layer_index]      # (B, T, H)
        
        
        # 各系列について EOS token の最後の出現位置を取る
        eos_mask = (input_ids == tokenizer.eos_token_id)

        for b in range(input_ids.size(0)):
            if LAST_TOKEN_IS_EOS:
                eos_positions = torch.where(eos_mask[b])[0]
                if len(eos_positions) == 0:
                    raise ValueError(f"EOS token が見つかりません: {batch_texts[b]}")
                eos_pos = eos_positions[-1].item()

                vec = layer_hs[b, eos_pos, :]   # (H,)
            else:
                seq_len = attention_mask[b].sum().item()
                vec = layer_hs[b, seq_len - 1, :]  # (H,)

            all_vecs.append(vec.detach().float().cpu().numpy())

    return np.stack(all_vecs, axis=0)


# =========================
# 特徴抽出
# =========================
features = extract_eos_hidden_states(
    proper_nouns,
    batch_size=BATCH_SIZE,
    layer_index=LAYER_INDEX
)


print("features shape:", features.shape)  # (N, hidden_dim)

# NaN / inf 除外
valid_mask = np.isfinite(features).all(axis=1)
valid_features = features[valid_mask]
valid_proper_nouns = [x for x, ok in zip(proper_nouns, valid_mask) if ok]
valid_categories = [x for x, ok in zip(categories, valid_mask) if ok]

print("valid samples:", len(valid_proper_nouns), "/", len(proper_nouns))

if len(valid_proper_nouns) < 2:
    raise ValueError("有効サンプルが少なすぎて PCA できません。")



# =========================
# PCA
# =========================
pca = PCA(n_components=2)
coords = pca.fit_transform(features)

print("Explained variance ratio:", pca.explained_variance_ratio_)



# =========================
# 色の割り当て
# =========================
unique_categories = list(dict.fromkeys(valid_categories))
cmap = plt.get_cmap("tab10")
category_to_color = {
    category: cmap(i % 10) for i, category in enumerate(unique_categories)
}

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 8))

for category in unique_categories:
    idxs = [i for i, c in enumerate(valid_categories) if c == category]

    xs = coords[idxs, 0]
    ys = coords[idxs, 1]
    labels = [valid_proper_nouns[i] for i in idxs]

    plt.scatter(xs, ys, label=category, color=category_to_color[category])

    for label, x, y in zip(labels, xs, ys):
        plt.text(x, y, label, fontsize=10)


plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA of EOS hidden states ({MODEL_NAME}, layer={LAYER_INDEX}, LAST_TOKEN_IS_EOS={LAST_TOKEN_IS_EOS})")
plt.grid(True)
plt.tight_layout()
# plt.show()
# 保存
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{MODEL_NAME.replace('/', '_')}_layer{LAYER_INDEX}_pca.png")
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")