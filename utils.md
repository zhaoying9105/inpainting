## save
保存checkpoint，默认model的name是'model'

## load
加载 checkpoint

## read_by_batch
一个python生成器，按批次读取数据
参数 label 数据是否包括 标签

1. data_size
表示单个图片的大小，因为是二进制存储的
```python
data_size = data_shape[0] * data_shape[1] * data_shape[2]
```

2. 如果数据包含label
真实的一个batch 就是 data_size + 1
```python
data_batch = np.fromfile(file_object, dtype=np.uint8,
                          count=(data_size + 1) * batch_size)
```
3. 如果数据包含label
就拆分为data 和 label
```python
data_batch = np.reshape(data_batch, (-1, data_size + 1))
images = np.reshape(
    data_batch[:, :data_size], (-1, data_shape[0], data_shape[1], data_shape[2]))
labels = data_batch[:, -1]
yield images, labels
```

## preprocess_image
图像预处理

参数：
- hidden_size 表示切割区域的大小
1. mask 表示 切割矩阵

2. masks_idx 表示切割开始的位置

3. 如果`random_block`为真，那么从图片随机选取位置切割掉`hidden_size`大小的区域
   否则，从图片中间切割掉`hidden_size`大小的区域

4. masked_images 表示切割后的图片


## combine_images
结合图像
1. 图像的数量
```python
num = images.shape[0]
```

## save_images
保存图片
1. 图片去中心化
```python
image = image * 127.5 + 127.5
```

2. 保存图片（不同的的通道数均可）


## compute_psnr_ssim
计算每对图片的 psnr 和 ssim 并返回均值

1. `compare_psnr` [峰值信噪比](https://zh.wikipedia.org/wiki/%E5%B3%B0%E5%80%BC%E4%BF%A1%E5%99%AA%E6%AF%94)
    
    峰值信噪比经常用作图像压缩等领域中信号重建质量的测量方法，它常简单地通过均方差（MSE）进行定义。两个m×n单色图像I和K，如果一个为另外一个的噪声近似，那么它们的的均方差定义为：


2. `compare_ssim` [结构相似性指标](https://zh.wikipedia.org/wiki/%E7%B5%90%E6%A7%8B%E7%9B%B8%E4%BC%BC%E6%80%A7)

    可以看成是失真影像的影像品质衡量指标，结构相似性的基本观念为自然影像是高度结构化的，亦即在自然影像中相邻像素之间有很强的关联性，而这样的关联性承载了场景中物体的结构资讯。人类视觉系统在观看影像时已经很习惯抽取这样的结构性资讯。因此，在设计影像品质衡量指标用以衡量影像失真程度时，结构性失真的衡量是很重要的一环。

## extend_array_by_index
扩展图片

inputs 输入的图片
index
full_height 扩展后的高度
full_width 扩展后的宽度