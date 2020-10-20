# Digit

1. Download MNIST and SVHN dataset accordingly.

2. Run

```python
train_num_semi.py --gpu $ --optimizer adam --learning_rate 0.001 --decay_step 50000 --pp 0.01 --log_dir log_digit
```

You may modify --pp to change the scale of the target domain involved in the training.