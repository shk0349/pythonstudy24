import tensorflow as tf
import keras
import urllib.request
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)

# url에서 파일 다운로드
url = "https://github.com/rickiepark/hg-mldl/raw/master/best-cnn-model.keras"
output_file = "best-cnn-model.keras"

model = keras.models.load_model('best-cnn-model.keras')
model.layers
model.summary()
print("=" * 50)

keras.utils.plot_model(model, show_shapes = True)

print("첫번째 합성곱 층의 가중치")
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)
print("=" * 50)

conv_weights = conv.weights[0].numpy()

print("가중치 배열의 평균 및 표준편차")
print(conv_weights.mean(), conv_weights.std())
print("=" * 50)

# 그래프 구성
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

fig, axs = plt.subplots(2, 16, figsize = (15, 2))
# 2 x 16 = 32개의 그래프 영역을 만들고 순서대로 커널 출력
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i * 16 + j], vmin = -0.5, vmax = 0.5)    # 0.0 ~ 0.31
        axs[i, j].axis('off')
# 배열의 마지막 차원을 순회하면서 (0 ~ 16) + j 번쨰까지의 가중치 값을 차례대로 출력
plt.show()

no_training_model = keras.Sequential()

# Conv2D층 1개 추가
no_training_model.add(keras.layers.Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))

# no_training_conv 변수에 Conv2D층 가중치 저장
no_training_conv = no_training_model.layers[0]

print("no_training_conv 변수의 Conv2D층 가중치")
print(no_training_conv.weights[0].shape)
print("=" * 50)

no_training_weights = no_training_conv.weights[0].numpy()
print("가중치 평균과 표준편차")
print(no_training_weights.mean(), no_training_weights.std())
print("=" * 50)

# 가중치 배열을 히스토그램으로 출력
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# .imshow() 함수를 이용하여 그림으로 출력
fig, axs = plt.subplots(2, 16, figsize = (15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:, :, 0, i * 16 + j], vmin = -0.5, vmax = 0.5)
        axs[i, j].axis('off')
plt.show()

conv_acti = keras.Model(model.inputs, model.layers[0].output)

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 첫번째 샘플 출력
plt.imshow(train_input[0], cmap = 'gray_r')
plt.show()

inputs = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_acti.predict(inputs)

print("maps 크기")
print(feature_maps.shape)
print("=" * 50)

fig, axs = plt.subplots(4, 8, figsize = (15, 8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, :, :, i * 8 + j])
        axs[i, j].axis('off')
plt.show()

fig, axs = plt.subplots(4, 8, figsize = (15, 8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[:, :, 0, i * 8 + j], vmin = -0.5, vmax = 0.5)
        axs[i, j].axis('off')
plt.show()

conv2_acti = keras.Model(model.inputs, model.layers[2].output)
feature_maps = conv2_acti.predict(train_input[0:1].reshape(-1, 28, 28, 1) / 255.0)

print("feature_maps 크기")
print(feature_maps.shape)
print("=" * 50)

model.summary()
print("=" * 50)

fig, axs = plt.subplots(8, 8, figsize = (12, 12))

for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0, :, :, i * 8 + j])
        axs[i, j].axis('off')
plt.show()