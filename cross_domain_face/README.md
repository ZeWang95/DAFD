# Cross Domain Face Recognition

1. Download initialization model from [link](https://arxiv.org/pdf/1909.11285.pdf). This model is obtained by projecting imagenet-pretrained VGG13 over Fourier-Bessel bases. It works not as good as the original imagenet-pretrained model, but is better than random initialization.

2. Modifying the path in train.py according to your data and model placements.

3. Run
'''
train.py --gpu # --lr 1e-2 --log_dir log_cross_domain_face
'''