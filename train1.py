

import tensorflow as tf
from dataset import train_data, dev_data, test_data
from dataset import tokenizer
from dataset import WeiBoDataGenerator
# from models import model
import settings
import matplotlib.pyplot as plt
import numpy as np
# 导入模型相关代码
EPSILON = 1e-07


def recall_m(y_true, y_pred):
    """
    计算召回率
    """
    y_true = tf.cast(y_true, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + EPSILON)
    return recall


def precision_m(y_true, y_pred):
    """
    计算精确率
    """
    y_true = tf.cast(y_true, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + EPSILON)
    return precision


def f1_m(y_true, y_pred):
    """
    计算f1值
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + EPSILON))


model = tf.keras.Sequential([
    tf.keras.layers.Input((None,)),
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

model.summary()

# 自定义回调函数，在每个 epoch 结束时进行测试并记录测试准确率
class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_generator):
        self.test_generator = test_generator
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        _, test_accuracy, _, _, _ = self.model.evaluate(self.test_generator.for_fit(),
                                                       steps=self.test_generator.steps)
        self.test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch + 1}: Test Accuracy = {test_accuracy:.4f}')

# 训练函数，可传入不同的损失函数、学习率和批量大小
def train_model(loss_function, learning_rate, batch_size):
    train_generator = WeiBoDataGenerator(train_data, tokenizer, batch_size)
    dev_generator = WeiBoDataGenerator(dev_data, tokenizer, batch_size)
    test_generator = WeiBoDataGenerator(test_data, tokenizer, batch_size)

    # 重新编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy', f1_m, precision_m, recall_m])

    # 自动保存模型
    checkpoint = tf.keras.callbacks.ModelCheckpoint(settings.BEST_WEIGHTS_PATH, monitor='val_f1_m', save_best_only=True,
                                                    save_weights_only=True, mode='max')

    # 创建自定义回调函数实例
    test_callback = TestAccuracyCallback(test_generator)

    # 训练
    history = model.fit(train_generator.for_fit(), steps_per_epoch=train_generator.steps, epochs=settings.EPOCHS,
                        validation_data=dev_generator.for_fit(), validation_steps=dev_generator.steps,
                        callbacks=[checkpoint, test_callback])

    return history, test_callback.test_accuracies

# 初始设置
# 'binary_crossentropy'
default_loss = tf.keras.losses.binary_crossentropy 
default_learning_rate = settings.LEARNING_RATE 
default_batch_size = settings.BATCH_SIZE

# （1）初始训练
history1, test_accuracies1 = train_model(default_loss, default_learning_rate, default_batch_size)
train_loss1 = history1.history['loss']
train_accuracy1 = history1.history['accuracy']

# 绘制初始训练的训练损失、训练准确率和测试准确率随 epoch 变化的图
epochs = range(1, settings.EPOCHS + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss1, 'b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracy1, 'r', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, test_accuracies1, 'g', label='Test Accuracy')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# （2）使用不同的损失函数进行训练
new_loss = 'hinge'
history2, test_accuracies2 = train_model(new_loss, default_learning_rate, default_batch_size)
train_loss2 = history2.history['loss']
train_accuracy2 = history2.history['accuracy']

# 绘制使用不同损失函数训练的训练损失、训练准确率和测试准确率随 epoch 变化的图
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss2, 'b', label='Training Loss')
plt.title('Training Loss (Hinge Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracy2, 'r', label='Training Accuracy')
plt.title('Training Accuracy (Hinge Loss)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, test_accuracies2, 'g', label='Test Accuracy')
plt.title('Test Accuracy (Hinge Loss)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# （3）使用不同的学习率进行训练
learning_rates = [0.1, 0.01, 0.001, 0.0001]

plt.figure(figsize=(12, 8))
for i, lr in enumerate(learning_rates):
    history, test_accuracies = train_model(default_loss, lr, default_batch_size)
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']

    plt.subplot(2, 2, i + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'g', label='Test Accuracy')
    plt.title(f'Learning Rate: {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    
# 确定纵坐标范围
ymin = 0
ymax = 1

# 生成间隔为 0.02 的刻度值
yticks = np.arange(ymin, ymax + 0.1, 0.1)
# 设置纵坐标刻度
plt.yticks(yticks)
plt.tight_layout()
plt.show()

# （4）使用不同的批量大小进行训练
batch_sizes = [16, 32, 64, 128]
plt.figure(figsize=(12, 10))
for i, bs in enumerate(batch_sizes):
    history, test_accuracies = train_model(default_loss, default_learning_rate, bs)
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    print('train_loss')
    print(train_loss)
    plt.subplot(3, 2, i + 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'g', label='Test Accuracy')
    plt.title(f'Batch Size: {bs}')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()


# 设置纵坐标刻度
plt.yticks(yticks)
plt.tight_layout()
plt.show()

# （5）可视化测试集前 100 个结果的预测标签、输入和实际标签
test_generator = WeiBoDataGenerator(test_data, tokenizer, default_batch_size)
predictions = model.predict(test_generator.for_fit(), steps=test_generator.steps)
predictions = (predictions > 0.5).astype(int)


actual_labels = []
batch_count = 0
for _, labels in test_generator.for_fit():
    print(f"Processing batch {batch_count + 1}, batch size: {len(labels)}")
    actual_labels.extend(labels)
    batch_count += 1
    if len(actual_labels) >= 100:
        break

actual_labels = actual_labels[:100]
predictions = predictions[:100]

inputs = []
batch_count = 0
for inputs_batch, _ in test_generator.for_fit():
    print(f"Processing input batch {batch_count + 1}, batch size: {len(inputs_batch)}")
    inputs.extend(inputs_batch)
    batch_count += 1
    if len(inputs) >= 100:
        break

plt.figure(figsize=(10, 10))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(inputs[i].reshape(1, -1), cmap='gray')
    plt.title(f'P: {predictions[i][0]}, A: {actual_labels[i][0]}', fontsize=6)
    plt.axis('off')


plt.tight_layout()
plt.show()
