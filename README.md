# Tensorflow Neural Machine Translation


### Training

1. **Preprocess the data**
```
python preprocess.py --out_dir /out/path
```

2. **Training**
```
python train.py --in_dir /input/path
```

3. **Predict**
```
python predict.py --model_path /model/path --source [source file]
```

4. **Eavluate**
```
python eval.py --model_path /model/path --source [source file]  --target [target file]
```
