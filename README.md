# multi_task_processes

This repository contains pytorch implementation of Multi-Task Processes.


### Basic Usage

**Training with missing rate 0.5**
```
python main.py --model [mtp/stp/jtp/mtp_s] --data [synthetic/weather] --gamma_train 0.5
```

**Testing with missing rate 0.5 and context size 10**
```
python test.py --eval_name [mtp/stp/jtp/mtp_s] --data [synthetic/weather] --gamma 0.5 --cs 10
```
