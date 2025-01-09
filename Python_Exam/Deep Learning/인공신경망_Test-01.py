import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()    # 버전 오류 해결

# 데이터 로드 및 훈련세트와 테스트 세트로 나누어 반환
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 패션잡화 이미지 출력
fig, axs = plt.subplots(1 ,10, figsize = (10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap = 'gray_r')
    axs[i].axis('off')
plt.show()

# 로지스틱 회귀로 패션 아이템 분류
train_scaled = train_input / 255.0
# SGDClassifier는 2차원배열을 지원하지 않으므로 1차원 배열로 변환
train_scaled = train_scaled.reshape(-1, 28 * 28)

sc = SGDClassifier(loss = 'log_loss', max_iter = 5)
scores = cross_validate(sc, train_scaled, train_target, n_jobs = -1)

print("SGDClassifier 적용한 5번 반복 테스트세트 평균")
print(np.mean(scores['test_score']))
print("=" * 50)

# 인공신경망(Artificial Neural Network / ANN) 이용한 패션아이템 분류로 문제성능 향상을 도모
    # 인공신경망의 z값을 계산하는 단위를 뉴런이라하며, 선형계산을 진행함
    # 처음 입력하는 값을 입력층(input_layer)이라고 함

# 딥러닝 라이브러리 중 TensorFlow 사용하여 연산
# 딥러닝 라이브러리는 GPU를 사용하여 인공신경망을 훈련함.
# GPU는 벡터와 행렬연산으로 3D연산에 최적화되어 있어 곱셈과 덧셈이 많이 수행되는 인공신경망의 속도 향상이 가능함
# Keras는 직접 GPU 연산을 수행하지는 않으나, GPU연산을 수행하는 라이브러리른 백엔드로 사용함
    # TensorFlow가 Keras를 백엔드로 사용가능 / 멀티-백엔드 케라스
    # Keras API만 익히면 다양한 딥러닝 라이브러리를 입맛대로 골라서 사용가능

# 인공신경망 모델 생성 / 교차검증을 사용하지 않고 검증세트를 별도로 덜어내어 사용
    # 1. 딥러닝 분야의 데이터셋이 충분히 크기 때문에 검증 점수가 안정적
    # 2. 교차 검증을 수행하기에는 훈련시간이 너무 오래 걸림(수시간 ~ 수일)

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2)

# 딥러닝에서는 다양한 종류의 층을 추가하는 식으로 모델을 구성
# Dense층을 사용하여 밀집층, 완전연결층(Fully Connected Layer) 구성 및 10개의 뉴런 사용
# 다중분류이므로 활성화함수는 Softmax 함수 사용, 입력값은 784개의 원소로 이뤄진 1차원 배열로 구성
    # 이진분류일 경우 Sigmoid 함수 사용

# Keras 첫번째 층 : 입력층 / 다음 추가되는 층은 자동으로 계산되므로 입력할 필요 없음
# Keras의 레이어 패키지 안에는 다양한 층이 존재, 가장 기본이 되는 층을 밀집층(Dense Layer)이라 함
    # 밀집층(Dense Layer) : 입력층과 뉴런이 모두 연결도니 선
# 이런 층을 뉴런이 모두 연결하고 있어 완전연결층(Fully Connected Layer)라고 부름

dense = keras.layers.Dense(10, activation = 'softmax', input_shape = (784, ))
# 입력값의 크기는 뉴런이 각각 몇개의 입력을 받는지 튜플로 지정

# 밀집층을 가진 신경망 모델 구성 시 Seqeuntial 클래스 사용
model = keras.Sequential([dense])
# Sequential 클래스의 객체를 만들때 앞에서 만든 밀집층 객체 dense를 전달
    # -> model 객체가 신경망 모델

# 활성화함수 : softmax와 같이 뉴런의 선형방정식 계산결과에 적용되는 함수

# 인공신경망으로 패션아이템 분류
# compile() 매서드 이용 keras 모델 훈련전 설정
# 손실함수 loss = 'sparse_categorical_crossentropy' (희소 다중분류 손실함수) : 정수타깃
#                'sparse_binary_crossentropy'(희소 이진분류 손실함수) -> 이진타깃
# 측정값 metrics = ['accuracy'] : 정확도(accuracy)의 지표가 됨
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# keras는 자동으로 epoch마다 손실을 기록함
# 원핫-인코딩 : 7을 0000000100 처럼 정수를 타깃처럼 출력하는것

print("keras 모델 훈련")
model.fit(train_scaled, train_target, epochs = 5)
print("=" * 50)

print("keras 모델 성능 평가")
model.evaluate(val_scaled, val_target)
print("=" * 50)