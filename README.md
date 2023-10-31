# Model Compression Practice
`np.packbits` 
<br></br><br><br><br>

## Vision Transformer
<br>Dataset: CIFAR-10
<br><br>Task: Image Classification
<br><br>Method: Fixed Point Quantization, Straight Through Estimator (STE)
<br><br>Experiment: Only quantized Multi-Layer-Percetron, Bitwidth (int8, int4, int2)

<br></br><br><br><br>

## BiBERT
<br>Dataset: SST-2
<br><br>Task: Natural Language Processing
<br><br>Method: Full Binarized Quantization, Straight Through Estimator (STE)
<br><br>Compression: numpy.packbits
<br></br><br><br><br>

## Reference
**Paper**
<br/>[Vaswani et al. Attention is All You Need. NeurIPS, 2017](https://arxiv.org/abs/1706.03762)
<br/>[Alexey et al. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR, 2021](https://arxiv.org/abs/2010.11929)
<br/>[Hubara et al. Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations, 2018](https://arxiv.org/abs/1609.07061)
<br/>[Yoshua Bengio et al. Estimating or propagating gradients through stochastic neurons for conditional computation. CoRR, abs/1308.3432, 2013](https://arxiv.org/abs/1308.3432)
<br/>[Alex Krizhevsky et al. Learning Multiple Layers of Features from Tiny Images. 2009.](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
<br/>[Vaswani et al. Attention is All You Need. NeurIPS, 2017](https://arxiv.org/abs/1706.03762)
<br/>[Alexey et al. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR, 2021](https://arxiv.org/abs/2010.11929)
<br/>[Haotong et al. BiBERT: Accurate Fully Binarized BERT, ICLR, 2022](https://arxiv.org/abs/2010.11929)

<br/>**Github**
<br/>[Huggingface Vision Transformers](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)
<br/>[Andrew-Tierno/QuantizedTransformer](https://github.com/Andrew-Tierno/QuantizedTransformer)
<br/>[htqin/BiBERT](https://github.com/htqin/BiBERT)
<br/>[Zhen-Dong/BitPack](https://github.com/Zhen-Dong/BitPack)
