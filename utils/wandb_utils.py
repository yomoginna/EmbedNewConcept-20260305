
import wandb
import os
project_root = os.path.join(os.path.dirname(__file__), "..")

# wandbの設定
def set_wandb_env(PROJECT_NAME, model_name_for_dirname, wandb_dir, WANDB_API_KEY):
    """wandbのloginと、学習結果記録に必要な設定を行う関数.
    これらはimport wandbの前に設定する必要がある.
    Args:
        - PROJECT_NAME: Wandbのプロジェクト名
        - model_name_for_dirname: モデル名（ディレクトリ名用）
        - wandb_dir: 学習結果記録を格納するディレクトリ
        - WANDB_API_KEY: WandbのAPIキー
    """
    print(f"Wandb Project name: {PROJECT_NAME}")
    os.makedirs(wandb_dir, exist_ok=True)

    tmp_root_path = os.path.join(project_root, "tmp")
    os.makedirs(tmp_root_path, exist_ok=True)

    os.environ["WANDB_MODE"] = "online"  # "disabled", "offline", "online"
    os.environ['TMPDIR'] = tmp_root_path
    os.environ['TEMP'] = tmp_root_path
    os.environ['TMP'] = tmp_root_path
    os.environ['DATA_DIR'] = tmp_root_path
    os.environ['ARTIFACT_DIR'] = tmp_root_path
    os.environ["WANDB_DATA_DIR"] = os.path.join(wandb_dir, ".wandb_data")

    os.environ['WANDB_DIR'] = wandb_dir
    os.environ['WANDB_CACHE_DIR'] = os.path.join(wandb_dir, ".wandb_cache")
    os.environ['WANDB_CONFIG_DIR'] = os.path.join(wandb_dir, ".wandb_config")
    os.environ['DATA_DIR'] = wandb_dir


    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    wandb_instance = None
    wandb_id = wandb.util.generate_id()  # 一意のIDを作成（再利用するため）
    run_name = f"TrainMemVec_{model_name_for_dirname}"
    wandb_instance = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        id=wandb_id,  # 一意のID
        resume="allow",
    )
    wandb_url = wandb_instance.get_url()
    print("WandB URL:", wandb_url)
