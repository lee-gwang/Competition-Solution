# 2019_Jeju_BigData_Competition
### "퇴근시간 버스승차인원 예측"
##### - 2nd place solution code(2nd/260)
##### - dacon homepage : https://dacon.io/rank13

## Description

### path
```
├── data(download dataset in dacon site)
│   ├── train.csv
│   ├── test.csv
│   └── submission_sample.csv
|
├── lgbm_bus.py ( train & inference)
│
├── dacon_function.py ( useful function)
|
└── submission
    └── lgb_model.csv(submission file)
```

## Example of usage
```
$> python lgbm_bus.py
```

## Requirements

- python 3.6
- numpy
- pandas
- tqdm
- lightgbm
