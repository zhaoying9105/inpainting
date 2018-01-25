# inpaint 

## 定义参数

- `dataname` 数据集的名字
- `mode` 模式，有 train validate test 三种
- `block` 图片裁剪模式，有 random center 两种
- `learning_rate` 学习率
- `lamb_rec` 图像重建的损失函数 的 权重
- `lamb_adv` 图像经过判别器的损失函数的 权重
- `lamb_tv` TV损失函数的 权重
- `use_l1` 使用L1 函数来衡量重建损失，如果是false就用L2
- `weight_clip` G D 的更新操作是是否进行 权重裁剪
- `image_size` 图像大小， 默认 128
- `image_channel` 图像通道
- `start_epoch` 开始的周期，默认是0
- `epoch` 训练的周期，默认是100
- `batch_size` 
- `ckpt` checkpoint 保存的频率，以周期记
- `sample` 样本保存的频率，以周期记
- `summary` 保存summary的 频率，以step计
- `gene_iter` 每个batch训练几次G ，默认是5
- `disc_iter` 每个batch训练几次，默认是1
- `gpu` GPU的 编号，0  1 2 这种


## adam算法的参数
```python
BETA1 = 0.5
BETA2 = 0.9
LAMB_REC = FLAGS.lamb_rec
LAMB_ADV = FLAGS.lamb_adv
LAMB_TV = FLAGS.lamb_tv
WEIGHT_DECAY_RATE = 0.00001
```

## run_model
函数非常长

1. masked_image_holder 裁剪后的图片
2. hidden_image_holder 可能是真实图片

> 注意  masked_image_holder 和 hidden_image_holder 大小不一样

3. 根据D 的输出判断准确率

```python
adv_real_score = discriminator(
      hidden_image_holder, BATCH_SIZE, is_train=TRAIN)
  adv_fake_score = discriminator(
      fake_image, BATCH_SIZE, reuse=True, is_train=TRAIN)
  adv_all_score = tf.concat([adv_real_score, adv_fake_score], axis=0)

  labels_disc = tf.concat(
      [tf.ones([BATCH_SIZE]), tf.zeros([BATCH_SIZE])], axis=0)
  labels_gene = tf.ones([BATCH_SIZE])
  correct = tf.equal(labels_disc, tf.round(tf.nn.sigmoid(adv_all_score)))
  disc_acc = tf.reduce_mean(tf.cast(correct, tf.float32))
```

4. 计算并合并三种loss，得到G 和D 的初步loss
    1. 重建loss
    2. 判别loss
    3. 方差loss， 方差loss 是生成的图片的像素点和相邻像素点之间的差值的绝对值的和，表示图像有多少噪音，减少这个loss可以压制图片里的噪音
```python
  if FLAGS.use_l1:
    rec_loss = tf.reduce_mean(
        tf.abs(tf.subtract(hidden_image_holder, fake_image)))
  else:
    rec_loss = tf.reduce_mean(
        tf.squared_difference(hidden_image_holder, fake_image))

  adv_disc_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_disc, logits=adv_all_score))
  adv_gene_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_gene, logits=adv_fake_score))

  tv_loss = tf.reduce_mean(tf.image.total_variation(fake_image))

  gene_loss_ori = LAMB_ADV * adv_gene_loss + \
      LAMB_REC * rec_loss + LAMB_TV * tv_loss
  disc_loss_ori = LAMB_ADV * adv_disc_loss
```

5. 对G和D里的所有权重，求他们的2范数，分别加到G 和D的loss（就是增加了正则项）
```python
  all_vars = tf.trainable_variables()
  gene_vars = [var for var in all_vars if 'generator' in var.name]
  disc_vars = [var for var in all_vars if 'discriminator' in var.name]
  gene_weights = [var for var in gene_vars if 'weights' in var.name]
  disc_weights = [var for var in disc_vars if 'weights' in var.name]
  gene_loss = gene_loss_ori + WEIGHT_DECAY_RATE * \
      tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in gene_weights]))
  disc_loss = disc_loss_ori + WEIGHT_DECAY_RATE * \
      tf.reduce_mean(tf.stack([tf.nn.l2_loss(x) for x in disc_weights]))
```

6. 对梯度进行修剪（clip）

如果使用梯度修剪，那么吧G和D的梯度都限制在 [-10,10]以内，
> 这里计算了梯度，但是没有进行参数更新，使用的 apply_gradients 函数是 minimize 函数的一个部分，真正的参数更新是在 minimize 函数中进行的
```python
  if TRAIN:
    if FLAGS.weight_clip:
      gene_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, BETA1)
      gene_vars_grads = gene_optimizer.compute_gradients(gene_loss, gene_vars)
      gene_vars_grads = [gv if gv[0] is None else [
          tf.clip_by_value(gv[0], -10., 10.), gv[1]] for gv in gene_vars_grads]
      gene_train_op = gene_optimizer.apply_gradients(gene_vars_grads)

      disc_optimizer = tf.train.AdamOptimizer(
          FLAGS.learning_rate / 10.0, BETA1)
      disc_vars_grads = disc_optimizer.compute_gradients(disc_loss, disc_vars)
      disc_vars_grads = [gv if gv[0] is None else [
          tf.clip_by_value(gv[0], -10., 10.), gv[1]] for gv in disc_vars_grads]
      disc_train_op = disc_optimizer.apply_gradients(disc_vars_grads)
```

7. 如果不必修剪就直接使用adam进行梯度下降

8. 做一些统计展示的工作
```python
    tf.summary.scalar('disc_acc', disc_acc)
    tf.summary.scalar('rec_loss', rec_loss)
    tf.summary.scalar('adv_gene_loss', gene_loss_ori)
    tf.summary.scalar('gene_loss', adv_gene_loss)
    tf.summary.scalar('disc_loss', adv_disc_loss)
    tf.summary.histogram('gene_score', adv_fake_score)
    tf.summary.histogram('disc_score', adv_all_score)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
```

9. 查看是否有checkpoint 可以 load

```python
  saver = tf.train.Saver(max_to_keep=1)
  could_load, checkpoint_counter = load(sess, saver, CHECKPOINT_DIR)
  if could_load:
    counter = checkpoint_counter
    print(' [*] Load SUCCESS')
  else:
    print(' [!] Load FAILED...')
    if not TRAIN:
      exit()
```

10. 如果是训练模式，就开始训练
    1. 获取图片并根据不同的裁剪选项进行预处理
    ```python
          file_object = open(DATA_PATH, 'rb')
      print('Current Epoch is: ' + str(epoch))
      for image_batch in read_by_batch(
              file_object, BATCH_SIZE, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]):
        if image_batch.shape[0] != BATCH_SIZE:
          break
        image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
        if FLAGS.block == 'center':
          masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
              image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL, False)
        else:
          masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
              image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL)
    ```
    2. 根据 G 产生的sample 对 裁剪后的图像进行补齐
    ```python
    inpaint_image[idx, idx_start1: idx_end1,
              idx_start2: idx_end2, :] = samples[idx, :, :, :]
    ```
    3. 保存图片
    ```python
    save_images(image_batch, index, 0, SAMPLE_DIR)
    save_images(inpaint_image, index, 1, SAMPLE_DIR)
    save_images(masked_image_batch, index, 2, SAMPLE_DIR)
    ```
    4. 保存summary文件
    ```python
            if counter % FLAGS.summary == 0:
          summary = sess.run(
              merged,
              feed_dict={
                  masked_image_holder: masked_image_batch,
                  hidden_image_holder: hidden_image_batch
              })
          writer.add_summary(summary, counter)
    ```
    5. 分别对D和G网络进行优化
    ```python
            for _ in range(FLAGS.disc_iter):
          _ = sess.run(
              disc_train_op,
              feed_dict={
                  masked_image_holder: masked_image_batch,
                  hidden_image_holder: hidden_image_batch
              })

        for _ in range(FLAGS.gene_iter):
          _, rec_loss_value, adv_gene_loss_value, adv_disc_loss_value, disc_acc_value = sess.run(
              [gene_train_op, rec_loss, adv_gene_loss, adv_disc_loss, disc_acc],
              feed_dict={
                  masked_image_holder: masked_image_batch,
                  hidden_image_holder: hidden_image_batch
              })
    ```
    6. 打印loss等信息

11. 如果不是训练模式
    1. 获取图片并中心化
    ```python
        for image_batch in read_by_batch(
            file_object, BATCH_SIZE, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]):
      if image_batch.shape[0] != BATCH_SIZE:
        break
      image_batch = (image_batch.astype(np.float32) - 127.5) / 127.5
    ```
    2. 裁剪图片
    ```python
          if FLAGS.block == 'center':
        masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
            image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL, False)
      else:
        masked_image_batch, hidden_image_batch, masks_idx = preprocess_image(
            image_batch, BATCH_SIZE, IMAGE_SIZE, HIDDEN_SIZE, IMAGE_CHANNEL)
    ```
    3. 保存补齐效果图
    ```python
          for idx in range(FLAGS.batch_size):
        idx_start1 = int(masks_idx[idx, 0])
        idx_end1 = int(masks_idx[idx, 0] + HIDDEN_SIZE)
        idx_start2 = int(masks_idx[idx, 1])
        idx_end2 = int(masks_idx[idx, 1] + HIDDEN_SIZE)
        inpaint_image[idx, idx_start1: idx_end1,
                      idx_start2: idx_end2, :] = samples[idx, :, :, :]
      save_images(image_batch, index, 0, SAMPLE_DIR)
      save_images(inpaint_image, index, 1, SAMPLE_DIR)
      save_images(masked_image_batch, index, 2, SAMPLE_DIR)
    ```
    4. 输出loss等参数


## save_configs
网络的超参数保存为txt文件

## print_args
打印网络的超参数

## main
主函数
1. 建必要的文件夹
2. 在gup上运行运算


