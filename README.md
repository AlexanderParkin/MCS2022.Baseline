# Baseline for MCS2022.Car models verification competition

This is a repository with a baseline solution for the MCS2022. Cars verification competition. In this competition, participants need to train a model to verify car models (models are the same, not the same car).

The idea of the basic solution is to train a classifier of car models, remove the classification layer and use embeddings to measure the proximity between two images.

## Steps for working with baseline
### 0. Download CompCars dataset
To train the model in this offline the CompCars dataset is used. You can download it [here](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html).

If you have problems getting box labels in datasets, you can use a duplicate of the labels that we posted here.
### 1. Prepare data for classification
Launch `prepare_data.py` to crop images on bboxes and generate lists for training and validation phases.
```bash
python prepare_data.py --data_path ./data/CompCars/ --annotation_path ./data/annotation/
```

### 2. Run model training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet18  \
                                      --root './data/image/' \
                                      --train_file ./data/annotation/train.txt \
                                      --val_file ./annotation/val.txt \
                                      --pretrained
```
### 3. Create a submission file