# convert_image

一些设置

- DATASET_NAME = 'voc_128' 数据集的名字
- IMAGE_SIZE = 128 图像大小
- IMAGE_CHANNEL = 3 图像通道

## display_bin


## convert_to_bin

将二进制数据文件转换成图片：
1. 读取文件
```python
  file_object = open('data/' + DATASET_NAME + '._train.bin', 'rb')
  #file_object = open('data/' + DATASET_NAME + '_test.bin', 'rb')
  images = np.fromfile(file_object, dtype=np.uint8)
```

2. 转换shape
```python
images = np.reshape(images, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL))
```
3. 判断channel
```python
  if IMAGE_CHANNEL == 1:
    plt.imshow(images[100, :, :, 0], cmap='gray')
  elif IMAGE_CHANNEL == 3:
    plt.imshow(images[10])
  else:
    print('image channel not supported')
```
4. 展示图片
```python
  plt.show()
```

## convert_to_bin
将图片转换成二进制文件
1. 要保存为jpg 文件
```python
path_pattern = dataset_path + '*.jpg'
```

2. 获取图片
因为 ImageCollection 是 字母顺序的， 所以用shuffle打乱

```python
images = np.array(io.ImageCollection(path_pattern))
np.random.shuffle(images)
```

3. 4：1 比例拆分为 训练集和测试集
```python
  train_num = num * 4 / 5
  image_train = images[:train_num]
  image_test = images[train_num:]
```

4. 保存为bin 文件
```python
  image_train.tofile('data/' + DATASET_NAME + '_train.bin')
  image_test.tofile('data/' + DATASET_NAME + '_test.bin')
```
