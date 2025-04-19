# Inference and Evaluation

### Prerequisite:

Install dependencies:

```
pip install -r requirements.txt
```

### Creating Synthetic Datasets:

```
python inference.py --modelPath=<PATH_TO_.PTH> --outputDir=<OUTPUT_DIR>
```

### Generating Metrics and t-SNE Charts:

```
python metrics-and-charts.py --genPath=<OUTPUT_DIR> --gtPath=<PATH_TO_GT> --trialName=<TRIAL_NAME>
```

### Training Downstream Classifiers:

```
python train_classifier.py --outputDir=<OUTPUT_DIR> --trainingPath=<PATH_TO_GENERATED_IMGS>
```