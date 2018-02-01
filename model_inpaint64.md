# model_inpaint128


## generator

生成器带有卷积层的自动编码器

先进行编码操作，经过6层卷积，从 shape = 64x64ximage_dim 到 潜变量 shape = 1 x 4000
在进行解码操作，经过5层转置卷积从 shape = 1 x 4000 到 图像 shape = 64 x 64 x image_dim 

## discriminator

判别器
经过5层卷积得到判别结果
> note : 判别器最后的线性层没有激活函数
