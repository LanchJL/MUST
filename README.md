# Mutual Balancing in State-Object Components for Compositional Zero-Shot Learning (MUST)
The code of Mutual Balancing in State-Object Components for Compositional Zero-Shot Learning (PR 2024)

## Datasets
The splits of dataset and its attributes can be found in utils/download_data.sh, the complete installation process can be found in [CGE] https://github.com/ExplainableML/czsl

Set the --DATA_FOLDER in flag.py as your dataset path.

## Train
If you wish to try training our model from scratch, please run train.py, for example:

```shell
  python train.py --config CONFIG_FILE
```

## Test
Please specify the path for the trained weights, and than run:

```shell
   python test.py --config CONFIG_FILE test_weights_path --WEIGHTS_PATH
```
