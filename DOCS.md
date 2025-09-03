# Setup
まず, 以下のコマンドを実行して環境を構築してください.
```shell
direnv allow # only for Nix env.
source ./.venv/bin/activate # needed if you don't have direnv
uv sync
```

次に, 以下のコマンドで必要なデータをダウンロードしてください. (かなり時間がかかると思います. )
```
wget https://zenodo.org/records/14342967/files/checkpoints.tar
wget https://zenodo.org/records/7884686/files/audioldm-s-full
```
また, [kaggleのdataset](https://www.kaggle.com/datasets/mashijie/eating-sound-collection)もダウンロードしてください. こちらはブラウザからのダウンロードが必要です.

checkpoints.tarは展開したうえで, 中身のファイルを`data/checkpoints/`直下に配置してください. audioldm-s-fullも同様に, `data/checkpoints`直下に配置してください.

kaggleのdatasetはダウンロードが完了したら, 圧縮ファイルを展開してリポジトリのルートディレクトリに配置してください.

# Create Dataset
Setupが完了したら, 以下のコマンドでカスタムデータセットを生成してください.
```shell
python preprocessor/execute.py --raw-dir ./clips_rd
```

# Finetune
Create Datasetが完了したら, 以下のコマンドでファインチューニングを実行してください.
```
python3 -m audioldm_train.train.latent_diffusion -c audioldm_train/config/kaggle_custom.yaml --reload_from_ckpt data/checkpoints/audioldm-s-full
```
