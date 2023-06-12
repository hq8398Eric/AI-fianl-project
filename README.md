# AI-fianl-project

### Overview
Datas including augumented ones are stored in data, the approches for data augmenting is in data_augmentation.

#### baseline
The baseline is for baseline code.

#### addpos
The addpos is for area added architecture.

#### match
The match is for switching data loader approch.

#### output_mod
The output_mod is for changing the output method.

### Code Usage
#### Train a model
```=bash
./script.sh
```

#### Generate predictions
```=bash
python generator.py {model_path}
```

#### Run evaluation metrics
- Both ground truth and prediction files are default in the `data` folder
```=bash
python evaluation.py
```
