# Cross Domain Face Recognition

1. Download initialization model from [link](https://drive.google.com/file/d/1ULOzYXGty_Psu9fVNpD8N-yK1phDksvK/view?usp=sharing). This model is obtained by projecting imagenet-pretrained VGG13 over Fourier-Bessel bases. It works not as good as the original imagenet-pretrained model, but is better than random initialization.

2. Modifying the path in train.py according to your data and model placements.

3. Run

```python
train.py --gpu $ --lr 1e-2 --log_dir log_cross_domain_face
```