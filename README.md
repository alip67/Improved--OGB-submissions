# Improved-Correct-and-Smooth-C-S-OGB-submissions


Paper: https://arxiv.org/abs/2010.13993

This directory contains OGB submissions. All hyperparameters were tuned on the validation set with optuna, except for products, which was hand tuned. All experiments were run with a RTX 2080 TI with 11GB.

## Some Tips 
- In general, the more complex and "smooth" your GNN is, the less likely it'll be that applying the "Correct" portion helps performance. In those cases, you may consider just applying the "smooth" portion, like we do on the GAT. In almost all cases, applying the "smoothing" component will improve performance. For Linear/MLP models, applying the "Correct" portion is almost always essential for obtaining good performance.

- In a similar vein, an improvement of performance of your model may not correspond to an improvement after applying C&S. Considering that C&S learns no parameters over your data, our intuition is that C&S "levels" the playing field, allowing models that learn interesting features to shine (as opposed to learning how to be smooth).
     - Even though GAT (73.57) is outperformed by GAT + labels (73.65), when we apply C&S, we see that GAT + C&S (73.86) performs better than GAT + labels + C&S (~73.70) , 
     - Even though a 6 layer GCN performs on par with a 2 layer GCN with Node2Vec features, C&S improves performance of the 2 layer GCN with Node2Vec features substantially more.
     - Even though MLP + Node2Vec outperforms MLP + Spectral in both arxiv and products, the performance ordering flips after we apply C&S.
     - On Products, the MLP (74%) is substantially outperformed by ClusterGCN (80%). However, MLP + C&S (84.1%) substantially outperforms ClusterGCN + C&S (82.4%).

- In general, autoscale works more reliably than fixedscale, even though fixedscale may make more sense...

## Arxiv

### Label Propagation (0 params):
```
python run_experiments.py --dataset arxiv --method lp

Valid acc: 0.7013658176448874
Test acc: 0.6832294302820814
```

### Plain Linear + C&S (5160 params, 52.5% base accuracy)
```
python gen_models.py --dataset arxiv --model plain --epochs 1000    
python run_experiments.py --dataset arxiv --method plain

Valid acc -> Test acc
Args []: 73.00 ± 0.01 -> 71.26 ± 0.01
```


###Our result:
```
C:\Python37\python.exe C:/Users/ALIP/Desktop/CorrectAndSmooth-master/CorrectAndSmooth-master/run_experiments.py --dataset arxiv --method plain
Original accuracy
All runs:
Highest Train: 55.1640 ± 0.0083
Highest Valid: 55.0408 ± 0.0091
  Final Train: 55.1640 ± 0.0083
   Final Test: 52.5035 ± 0.0172
Valid: 0.7300244974663579, Test: 0.7126309075571466
Valid: 0.7299573811201718, Test: 0.7126514824187807
Valid: 0.7299238229470788, Test: 0.7125074583873423
Valid: 0.7299238229470788, Test: 0.7125691829722445
Valid: 0.7300244974663579, Test: 0.7126514824187807
Valid: 0.7298902647739857, Test: 0.7125691829722445
Valid: 0.7297560320816134, Test: 0.7126720572804148
Valid: 0.7298567066008926, Test: 0.7126514824187807
Valid: 0.7299238229470788, Test: 0.7126103326955127
Valid: 0.7299238229470788, Test: 0.7126514824187807
Valid acc -> Test acc
Args []: 72.99 ± 0.01 -> 71.26 ± 0.01
```
### Linear + C&S (15400 params, 70.11% base accuracy)
```
python gen_models.py --dataset arxiv --model linear --use_embeddings --epochs 1000 
python run_experiments.py --dataset arxiv --method linear

Valid acc -> Test acc
Args []: 73.68 ± 0.04 -> 72.22 ± 0.02;
```


### Our result: 
'''
Original accuracy
All runs:
Highest Train: 74.7574 ± 0.1020
Highest Valid: 71.3886 ± 0.0397
  Final Train: 74.7574 ± 0.1020
   Final Test: 70.4422 ± 0.0220
Valid: 0.7370381556428068, Test: 0.723494434499928
Valid: 0.737776435450854, Test: 0.7244614529967286
Valid: 0.7370717138158999, Test: 0.7242762792420221
Valid: 0.7370381556428068, Test: 0.7238236322860728
Valid: 0.7380113426625055, Test: 0.7248318005061416
Valid: 0.7373066210275513, Test: 0.723515009361562
Valid: 0.7373066210275513, Test: 0.7237001831162685
Valid: 0.7369710392966207, Test: 0.7234532847766598
Valid: 0.7365012248733179, Test: 0.7231858115754172
Valid: 0.7370717138158999, Test: 0.7236384585313663
Valid acc -> Test acc
Args []: 73.72 ± 0.04 -> 72.38 ± 0.05
'''

### MLP + C&S (175656 params, 71.44% base accuracy)
```
python gen_models.py --dataset arxiv --model mlp --use_embeddings
python run_experiments.py --dataset arxiv --method mlp

Valid acc -> Test acc
Args []: 73.91 ± 0.15 -> 73.12 ± 0.12
```

### GAT + C&S (1567000 params, 73.56% base accuracy)
```
cd gat && python gat.py --use-norm
cd .. && python run_experiments.py --dataset arxiv --method gat

Valid acc -> Test acc
Args []: 74.84 ± 0.07 -> 73.86 ± 0.14
```

### Notes
As opposed to the paper's results, which only use spectral embeddings, here we use spectral *and* diffusion embeddings, which we find improves Arxiv performance.

## Products

### Label Propagation (0 params):
```
python run_experiments.py --dataset products --method lp 

Valid acc:  0.9090608549703736
Test acc: 0.7434145274640762
```

### Plain Linear + C&S (4747 params, 47.73% base accuracy)
```
python gen_models.py --dataset products --model plain --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method plain

Valid acc -> Test acc
Args []: 91.03 ± 0.01 -> 82.54 ± 0.03
```

### Linear + C&S (10763 params, 50.05% base accuracy)
```
python gen_models.py --dataset products --model linear --use_embeddings --epochs 1000 --lr 0.1
python run_experiments.py --dataset products --method linear

Valid acc -> Test acc
Args []: 91.34 ± 0.01 -> 83.01 ± 0.01
```

### MLP + C&S (96247 params, 63.41% base accuracy)
```
python gen_models.py --dataset products --model mlp --hidden_channels 200 --use_embeddings
python run_experiments.py --dataset products --method mlp

Valid acc -> Test acc
Args []: 91.47 ± 0.09 -> 84.18 ± 0.07
```
