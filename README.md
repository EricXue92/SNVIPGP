To install requirements:
```setup
pip install -r requirements.txt
```

Run example:

The running results are automatically saved as .csv files in the /results/conformal or /results/normal directory.

If you want to train SNVIPGP, add the flag (adjust the datasets and parameters):
```setup
python train.py --sniggp --dataset "CIFAR10" --OOD "SVHN" --conformal_training --spectral_normalization --inducing_point 15 --learning_rate 0.05 --epoch 50
```

If you want to train SNGP, simply add the flag :
```setup
python train.py --SNGP 
```

