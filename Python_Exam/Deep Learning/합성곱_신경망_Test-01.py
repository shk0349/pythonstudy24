import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Fashion MNIST를 활용한 케라스 합성곱층 응용
tf.keras.utils.set_random_seed(2)

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2)

# 합성곰 신경망 생성
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
print("Conv2D층 추가 - 1")
model.summary()
print("=" * 50)

model.add(keras.layers.MaxPooling2D(2))
print("maxpooling 적용 - 1")
model.summary()
print("=" * 50)

model.add(keras.layers.Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
print("Conv2D층 추가 - 2")
model.summary()
print("=" * 50)

model.add(keras.layers.MaxPooling2D(2))
print("maxpooling 적용 - 2")
model.summary()
print("=" * 50)

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation = 'softmax'))

print("Flatten, 은닉층, Dropout, 출력층 추가")
model.summary()
print("=" * 50)

# keras 층 구성을 그림으로 출력
keras.utils.plot_model(model)

# + 입력과 출력 크기 표시
keras.utils.plot_model(model, show_shapes = True)

# Compile 및 훈련시작

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras', save_best_only = True)

early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)

history = model.fit(train_scaled, train_target, epochs = 20, validation_data = (val_scaled, val_target), callbacks = [checkpoint_cb, early_stopping_cb])
print("=" * 50)

# 그래프 구성
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()

# 성능평가
model.evaluate(val_scaled, val_target)
print("=" * 50)

# 테스트세트로 성능평가
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)
print("=" * 50)

# 이미지 출력
plt.imshow(val_scaled[50].reshape(28, 28), cmap = 'gray_r')
plt.show()

# 확률 예측
preds = model.predict(val_scaled[50:51])
print(preds)

# 그래프화
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '앵클부츠']
print(classes[np.argmax(preds)])
