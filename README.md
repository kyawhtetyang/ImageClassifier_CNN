# ImageClassifier_CNN (v0)

PyTorch image classifier with:
- Transfer learning (MobileNetV2) or a small custom CNN
- Simple training + evaluation pipeline
- CLI prediction

## Expected Layout

This repo expects a dataset at `v0/data/` with this structure:

```text
data/
  train/
    class_1/
    class_2/
  test/
    class_1/
    class_2/
```

Recommended approach (keeps data out of snapshots/git):

1. Put data in pCloud: `~/pcloud/02_Execution/06_ImageClassifier_CNN/data/...`
2. Link it into the execution workspace:

```bash
cd ~/execution/06_Data_and_AI/06_ImageClassifier_CNN
ln -s ~/pcloud/02_Execution/06_ImageClassifier_CNN/data data
cd v0
ln -s ../data data
```

## Run

```bash
cd ~/execution/06_Data_and_AI/06_ImageClassifier_CNN/v0
conda activate pt
pip install -r requirements.txt
python main.py
python predict.py data/test/class_1/example.jpg
```

Adjust training parameters in `config.json`.
