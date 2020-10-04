# pMRI-Net

The project 'pMRI-Net-ZF' implements the method described in "Deep Parallel MRI Reconstruction Network Without Coil Sensitivities"
"https://arxiv.org/pdf/2008.01410.pdf"

Test data, learned weights and sampling mask can be downloaded in https://drive.google.com/drive/folders/1jVV0qk_4iZlY10wKadQiAEP4Sr1V6Z9N?usp=sharing
File named 'pd_Phase5_ZF' is learned weights for 'pMRI-Net-ZF.py'
File named 'pd_Phase4_K_c' is learned weights for 'pMRI-CNet-K.py'

# Requirements

The code was implementated on Window 10 via tensorflow-gpu 1.10.0, python 3.6.10

# Training

For training the network, simply use

```python pMRI-CNet-ZF.py```
or
```python pMRI-CNet-K.py```


# Testing

The reconstruction process is automatically start after training process stopped for certain epochs.
We provided the learned weights that has already trained, you can just delete the training part in the code.
The output is ```*.mat``` file
