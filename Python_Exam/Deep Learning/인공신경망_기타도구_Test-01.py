import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 인공신경망을 구성하는 기본절차
    # 1. 더미데이터 준비
    # 2. 훈련용, 검증용, 테스트용으로 데이터 분리
    # 3. 심층 구성(dense)
    # 4. model에 적용
    # 5. compile 진행(최적화 기법 적용 / adam)
    # 6. fit(훈련)
    # 7. evaluate(검증)

# 손실곡선 : fit() 메서드로 모델을 훈련하는 동안 훈련과정을 상세하게 출력 및 확인가능
    # -> epoch 횟수, 손실, 정확도 등 메세지 출력
    # fit() 메서드는 History 라는 클래스 객체 반환 / 객체 내에는 훈련과정의 지표, 손실, 정확도 등이 내장
        # -> 이 값을 잘 활용하면 그래프 작성 가능
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)

def model_fn(a_layer = None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape = (28, 28)))
    model.add(keras.layers.Dense(100, activation = 'relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation = 'softmax'))
    return model

model = model_fn()
print("model 요약정보")
model.summary()
print("=" * 50)

print("model compile")
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("=" * 50)

# fit() 메서드 결과를 history에 담음
history1 = model.fit(train_scaled, train_target, epochs = 5, verbose = 1)
print("=" * 150)

# 손실율 및 정확도 출력 / 5회 반복
plt.plot(history1.history['loss'])
plt.plot(history1.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'accuracy'])
plt.show()

# 손실율 및 정확도 출력 / 5회 반복
model2 = model_fn()
model2.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("=" * 150)
history2 = model2.fit(train_scaled, train_target, epochs = 20, verbose = 1)
plt.plot(history2.history['loss'])
plt.plot(history2.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'accuracy'])
plt.show()

# 검증손실
# 확률적 경사 하강법을 사용하였을 때 과대/과소 적합과 에포크 사이에 관계 부분이 존재하였음
# 인공신경망은 모두 일종의 경사 하강법을 사용하기 때문에 동일한 개념이 여기에도 적용됨

# 에포크에 따른 과대적합, 과소적합을 파악하려면 훈련세트 점수와 검증세트 점수에 대한 점수도 필요함
# 그래서 위처럼 훈련세트의 손실과 정확도만 그리면 파악이 어려움
# 때문에 검증 손실을 이용하여 과대/과소적합을 응용하면 됨

# keypoint
# 손실을 사용하는 것과 정확도를 사용하는것의 차이점 확인
# 인공신경망 모델이 최적화하는 대상은 정확도가 아니라 손실함수이고,
# 손실 감소에 비례하여 정확도가 높아지지 않는 경우도 존재하는데,
# 모델이 잘 훈련되었는지 판단하려면 정확도보다는 손실함수의 값을 확인하는 것이 더 좋은 방법임

# 테스트세트 및 검증세트의
model3 = model_fn()
model3.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("=" * 150)
history3 = model3.fit(train_scaled, train_target, epochs = 20, verbose = 1, validation_data = (val_scaled, val_target))
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# RMSprop 적용
model4 = model_fn()
model4.compile(optimizer = 'RMSprop', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("=" * 150)
history4 = model4.fit(train_scaled, train_target, epochs = 20, verbose = 1, validation_data = (val_scaled, val_target))
plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.plot(history4.history['accuracy'])
plt.plot(history4.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# Adagrad 적용
model5 = model_fn()
model5.compile(optimizer = 'Adagrad', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("=" * 150)
history5 = model5.fit(train_scaled, train_target, epochs = 20, verbose = 1, validation_data = (val_scaled, val_target))
plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.plot(history5.history['accuracy'])
plt.plot(history5.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# Adam 적용
model6 = model_fn()
model6.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("=" * 150)
history6 = model6.fit(train_scaled, train_target, epochs = 20, verbose = 1, validation_data = (val_scaled, val_target))
plt.plot(history6.history['loss'])
plt.plot(history6.history['val_loss'])
plt.plot(history6.history['accuracy'])
plt.plot(history6.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

# Dropout(드롭아웃) : 딥러닝의 기초가 되는 제프리 힌턴이 소개한 것으로 훈련과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서
#                     (뉴런 출력을 0으로 만듦) 과대적합을 막는것

# 어떤 샘플을 처리할 때 은닉층의 두번째 뉴런이 드롭아웃되어 h2 출력이 없음
# 다른 샘플을 처리할 때는 은닉층의 첫번쨰 뉴런이 드롭아웃되어 h1 출력이 없음
# 뉴런은 랜덤하게 드롭아웃되고 얼마나 많은 뉴런을 드롭할지 우리가 하이퍼파라미터에 정함

# 드롭아웃이 왜 과대적합을 막을 수 있을지 생각을 해보면 이전 층의 일부 뉴런이 랜덤하게 꺼지면
# 특정 뉴런에 과대하게 의존하는 것을 감소시킬 수 있고, 모든 입력에 대해 주의를 기울여야함
# 일부 뉴런의 출력이 없을 수 있다는 것을 감안하면, 이 신경망은 더 안정적인 예측을 만들 수 있다는 것임

# 또 다른 분석
# 앞의 드롭아웃이 적용된 2개의 신경망 그림을 보면 드롭아웃을 적용하여 훈련하는 것은
# 마치 2개의 신경망을 앙상블하는 것처럼 상상하게끔 함
# 앙상블 : 더 좋은 예측을 만들기 위해 여러 개의 모델을 3훈련하는 머신런닝의 알고리즙으로
# 과대적합을 막아주는 좋은 기법임

# 케라스에서 드롭아웃을 keras.layers.Dropout으로 제공하며, 어떤 층의 뒤에 드롭아웃을 두어 이 층의 출력을 랜덤하게 0으로 만듦
# 드롭아웃이 층처럼 사용되지만 훈련되는 파라미터는 없음

# model_fn 함수에 드롭아웃 객체를 전달하여 층을 추가함
model = model_fn(keras.layers.Dropout(0, 3))    # 30%정도 드롭아웃
print("model 요약")
model.summary()
print("=" * 50)
# 요악확인
# 은닉층 뒤에 추가된 드롭아웃 층(3번째 값)은 훈련되는 모델 파라미터가 없음
# 입력과 출력의 크기가 같으며, 일부 뉴런의 출력을 0으로 만들지만 전체 출력 배열의 크기를 바꾸지는 않음

# model 저장 및 복원
model7 = model_fn(keras.layers.Dropout(0, 3))
model7.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history7 = model7.fit(train_scaled, train_target, epochs = 10, verbose = 0, validation_data = (val_scaled, val_target))
model7.save('model-whole.keras')    # 모델 저장
model7.save_weights('model.weights.h5')    # HDF5 형식으로 저장
plt.plot(history7.history['loss'])
plt.plot(history7.history['val_loss'])
plt.plot(history7.history['accuracy'])
plt.plot(history7.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

import os

# 현재 디렉토리에서 'model'로 시작하는 파일 리스트 출력
for file in os.listdir('.'):
    if file.startswith('model'):
        print(file)

# Test-1
# 훈련하지 않은 새로운 모델을 만들고 model.weights.h5 파일에서 훈련된 모델 파라미터를 읽어 사용

# model_fn() 위와 동일한 모델 구성
model8 = model_fn(keras.layers.Dropout(0, 3))

model8.load_weights('model.weights.h5')
# 이때 사용하는 메서드는 save_weights()와 쌍을 이루는 load_weight() 메서드임
    # 주의사항 : load_weight() 메서드를 사용하려면
    #            save_weight() 메서드로 저장하였던 모델과 정확히 같은 구조를 가져야함

print("model 요약")
model8.summary()
print("=" * 50)

# 모델의 검증 정확도 확인
val_labels = np.argmax(model8.predict(val_scaled), axis = -1)
print("정확도 출력")
print(np.mean(val_labels == val_target))
print("=" * 50)

# 모델 전체에서 파일을 읽은 후 검증세트의 정확도 출력
model9 = keras.models.load_model('model-whole.keras')
print("검증세트의 정확도")
model9.evaluate(val_scaled, val_target)
print("=" * 50)

# callback(콜백) : 훈련과정 중간에 어떤 작업을 수행할 수 있게 하는 객체 / keras.callback 패키지에 있음
    # fit() 메서드의 callback 매개변수에 리스트로 전달하여 사용
# ModelCheckpoint callback을 사용할 예정이며, 에포크마다 모델을 저장함
# save_best_only = True : 매개변수를 지정, 가장 낮은 검증점수를 만드는 모델 저장가능
model10 = model_fn(keras.layers.Dropout(0, 3))
model10.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras', save_best_only = True)
model10.fit(train_scaled, train_target, epochs = 20, verbose = 1, validation_data = (val_scaled, val_target), callbacks = [checkpoint_cb])

# 모델 훈련 후 best-model.keras에 최상의 검증점수를 낸 모델이 저장됨
model10 = keras.models.load_model('best-model.keras')
print("best-model.keras load 검증세트의 정확도")
model10.evaluate(val_scaled, val_target)
print("=" * 50)

# 케라스에는 조기종료를 위한 EarlyStopping 콜백을 제공함
# 이 콜백의 patience 매개변수는 검증 점수가 향상되지 않더라도 참을 에포크 횟수로 지정
# patience = 2 : 2번연속 검증점수가 향상되지 않으면 훈련을 중지(patience : 인내심)
# restore_best_weights = True : 가장 낮은 검증 손실을 낸 모델 파라미터로 구동

model11 = model_fn(keras.layers.Dropout(0, 3))
model11.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras', save_best_only = True)
# 가장 낮은 검증 손실의 모델을 파일에 저장하고 검증 손실이 다시 상승할 때 훈련을 중지할 수 있음
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)
# 훈련을 중지하고 현재의 파라미터를 최상의 파라미터로 되돌림
history11 = model11.fit(train_scaled, train_target, epochs = 20, verbose = 1, validation_data = (val_scaled, val_target), callbacks = [checkpoint_cb, early_stopping_cb])
print("=" * 150)
plt.plot(history11.history['loss'])
plt.plot(history11.history['val_loss'])
plt.plot(history11.history['accuracy'])
plt.plot(history11.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'])
plt.show()

print("조기종료 시점 출력")
print(early_stopping_cb.stopped_epoch)
print("=" * 50)

print("조기종료 적용 시 검증결과")
model11.evaluate(val_scaled, val_target)
print("=" * 50)