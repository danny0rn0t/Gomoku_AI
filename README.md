# Gomoku AI

## Introduction
An implementation of gomoku AI base on [AlphaGo Zero](https://www.nature.com/articles/nature24270.pdf)

## Basic Usage
Train the Gomoku AI
```shell
python3 main.py --train
```

Play with the trained 9x9 board Gomoku AI
```shell
python3 main.py --play --boardsize=9 -p1 HUMAN -p2 AI --time_limit=10
```
Play with the trained 15x15 board Gomoku AI
- Note that 15x15 Gomoku is not well trained due to lack of computation resources

```shell
python3 main.py --play --boardsize=15 --num_layer=10 -p1 HUMAN -p2 AI --time_limit=10
```

Play with GUI support (require pygame)
``shell
python3 main.py --play --boardsize=9 -p1 HUMAN -p2 AI --time_limit=10 --GUI
```

Check out other options in main.py