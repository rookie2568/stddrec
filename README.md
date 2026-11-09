## ***This is the codebase of STDDRec***

STDDRec works with the following operating systems:

* Linux
* Windows 10
* macOS X

STDDRec requires Python version 3.7 or later.

Our core code located in recbole/model/sequential_recommender/STDDRec



STDDRec requires torch version 1.7.0 or later. If you want to use STDDRec with GPU, please ensure that CUDA or cudatoolkit version is 9.2 or later. This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

With the source code, you can use the provided script for initial usage of our library:

```shell
python run_recbole.py
```

This script will run the BPR model on the ml-100k dataset.

Typically, this example takes less than one minute. We will obtain some output like:

```
INFO ml-100k
The number of users: 944
Average actions of users: 106.04453870625663
The number of items: 1683
Average actions of items: 59.45303210463734
The number of inters: 100000
The sparsity of the dataset: 93.70575143257098%
INFO Evaluation Settings:
Group by user_id
Ordering: {'strategy': 'shuffle'}
Splitting: {'strategy': 'by_ratio', 'ratios': [0.8, 0.1, 0.1]}
Negative Sampling: {'strategy': 'full', 'distribution': 'uniform'}
INFO BPRMF(
    (user_embedding): Embedding(944, 64)
    (item_embedding): Embedding(1683, 64)
    (loss): BPRLoss()
)
Trainable parameters: 168128
INFO epoch 0 training [time: 0.27s, train loss: 27.7231]
INFO epoch 0 evaluating [time: 0.12s, valid_score: 0.021900]
INFO valid result:
recall@10: 0.0073  mrr@10: 0.0219  ndcg@10: 0.0093  hit@10: 0.0795  precision@10: 0.0088
...
INFO epoch 63 training [time: 0.19s, train loss: 4.7660]
INFO epoch 63 evaluating [time: 0.08s, valid_score: 0.394500]
INFO valid result:
recall@10: 0.2156  mrr@10: 0.3945  ndcg@10: 0.2332  hit@10: 0.7593  precision@10: 0.1591
INFO Finished training, best eval result in epoch 52
INFO Loading model structure and parameters from saved/***.pth
INFO best valid result:
recall@10: 0.2169  mrr@10: 0.4005  ndcg@10: 0.235  hit@10: 0.7582  precision@10: 0.1598
INFO test result:
recall@10: 0.2368  mrr@10: 0.4519  ndcg@10: 0.2768  hit@10: 0.7614  precision@10: 0.1901
```

If you want to change the parameters, such as `learning_rate`, `embedding_size`, just set the additional command parameters as you need:

```shell
python run_recbole.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```shell
python run_recbole.py --model=[model_name]
```



