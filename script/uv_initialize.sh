
# source script/uv_initialize.sh
uv python install 3.12
uv python pin 3.12
uv init

uv add accelerate adjusttext google-genai ipykernel jupyterlab language-tool-python transformers pandas numpy \
    dataset tqdm matplotlib scikit-learn wandb python-dotenv dotenv nltk seaborn japanize-matplotlib notebook \
    plotnine pytorch-triton sentence-transformers sentencepiece sparqlwrapper requests openai google-genai \
    wikipedia-api wikipedia2vec pyyaml
# torchのみピンポイントで指定する必要があるので，以下のようにuv add でtomlに追加してから，uv syncでuv環境にインストールする（そうしないと，torchだけversionが合わなくてインストールできないと言われるので）
uv add "torch>=2.10.0.dev20251029" \
  --index pytorch-cu128-nightly=https://download.pytorch.org/whl/nightly/cu128
uv sync

# いらなくなったpackage
# gensim