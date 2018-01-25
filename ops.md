# ops

## concat
连接操作

## batch_norm
批正则化

## conv_cond_concat
参数
input_x 输入数据
input_y 条件向量

1. 把y 扩展到x 的shape
```python
input_y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])
```
2. x 和y 在通道轴叠加，就是x 和 y 是两个图片，这个操作就是两个图层叠加起来

```python
concat([input_x, input_y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
```


## conv2d
卷积层
固定了stride = 2，kernel = 2，padding = "same"
加上bais

## transconv2d
转置卷积层
固定了stride = 2，kernel = 2，padding = "same"
加上bais

## lrelu
leaky relu

## prelu
parameter relu

## pooling
池化层
固定了stride = 2，kernel = 2，padding = "same"

## fully_connect
普通的全连接层

## channel_wise_fc
全连接层，对不同的channel均适用
经过reshape操作，本来 [batch_size,width,height,channel]转变为[channel,batch_size,width * height]
再经过全连接层，这就要求在一个网络中，batch_size 是不变的
```pyhton
  _, width, height, channel = inputs.get_shape().as_list()
  input_reshape = tf.reshape(inputs, [-1, width * height, channel])
  input_transpose = tf.transpose(input_reshape, [2, 0, 1])
```