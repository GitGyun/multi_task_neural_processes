# Multi-Task Neural Processes

This repository contains a pytorch implementation of [Multi-Task Neural Processes](https://arxiv.org/abs/2110.14953).


## Basic Usage

### Training with missing rate 0.5
```
python main.py --model [mtp/stp/jtp/mtp_s] --data [synthetic/weather] --gamma_train 0.5
```

### Testing with missing rate 0.5 and context size 10
```
python test.py --eval_name [mtp/stp/jtp/mtp_s] --data [synthetic/weather] --gamma 0.5 --cs 10
```

## Citation
If you find this work useful, please consider citing:
```bib
@inproceedings{kim2021multi,
  title={Multi-Task Processes},
  author={Kim, Donggyun and Cho, Seongwoong and Lee, Wonkwang and Hong, Seunghoon},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
