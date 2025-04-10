
import tensorflow as tf
from dataset import train_data, dev_data, test_data
from dataset import tokenizer
from dataset import WeiBoDataGenerator
from models import model
import settings
import matplotlib.pyplot as plt

class TestAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_generator):
        self.test_generator = test_generator
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        _, test_accuracy, _, _, _ = self.model.evaluate(self.test_generator.for_fit(),
                                                       steps=self.test_generator.steps)
        self.test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch + 1}: Test Accuracy = {test_accuracy:.4f}')

train_generator = WeiBoDataGenerator(
    train_data, tokenizer, settings.BATCH_SIZE)
dev_generator = WeiBoDataGenerator(dev_data, tokenizer, settings.BATCH_SIZE)
test_generator = WeiBoDataGenerator(test_data, tokenizer, settings.BATCH_SIZE)

# 创建自定义回调函数实例
test_callback = TestAccuracyCallback(test_generator)

# 自动保存模型
checkpoint = tf.keras.callbacks.ModelCheckpoint(settings.BEST_WEIGHTS_PATH, monitor='val_f1_m', save_best_only=True,
                                                save_weights_only=True, mode='max')
# 训练
history = model.fit(train_generator.for_fit(), steps_per_epoch=train_generator.steps, epochs=settings.EPOCHS,
                    validation_data=dev_generator.for_fit(), validation_steps=dev_generator.steps,
                    callbacks=[checkpoint, test_callback])
# 记录训练过程中的损失和准确率
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
test_accuracies = test_callback.test_accuracies
# model.fit_generator(train_generator.for_fit(), steps_per_epoch=train_generator.steps, epochs=settings.EPOCHS,
#                     validation_data=dev_generator.for_fit(), validation_steps=dev_generator.steps,
#                     callbacks=[checkpoint, ])
# 测试
print('test set results：')
loss, accuracy, f1_score, precision, recall = model.evaluate(test_generator.for_fit(),
                                                                       steps=test_generator.steps)
# loss, accuracy, f1_score, precision, recall = model.evaluate_generator(test_generator.for_fit(),
#                                                                        steps=test_generator.steps)
print('loss =', loss)
print('accuracy =', accuracy)
print('f1 score =', f1_score)
print('precision =', precision)
print('recall =', recall)


# 绘制训练损失、训练准确率和测试准确率随 epoch 变化的图
epochs = range(1, settings.EPOCHS + 1)

plt.figure(figsize=(12, 4))

# 绘制训练损失曲线
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练准确率曲线
plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 绘制测试准确率曲线
plt.subplot(1, 3, 3)
plt.plot(epochs, test_accuracies, 'g', label='Test Accuracy')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

