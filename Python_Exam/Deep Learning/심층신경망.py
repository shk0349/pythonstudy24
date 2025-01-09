import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# keras API에서 패션MNIST 데이터셋을 불러와 훈련세트와 테스트세트 구성
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0    # 이미지 픽셀값을 0 ~ 1 사이로 변환
train_scaled = train_scaled.reshape(-1, 28 * 28)    # 1차원배열로 변환

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2)

# 인공신경망 모델에 2개의 layer 추가하여 연산
    # 입력층과 출력층 사이에 밀집층이 추가되며, 이를 은닉층(hidden layer)이라 함
# 활성화함수 : 신경망 층의 선형방정식 계산값에 적용하는 함수
# 출력층에 적용하는 활성화 함수 종류는 제한적
    # 이진분류 : 시그모이드함수, 다중분류 : 소프트맥스함수
# 은닉층은 활성화 함수 사용이 자유로움(시그모이드 함수, 렐루함수 등)

# 신경망도 마찬가지로 은닉층에서 선형적인 산술계산만 수행한다면 수행할 역할이 없는 셈이니,
# 선형계산을 적당히 비선형적으로 비틀어줘야만 다음 층의 계산과 단순히 합쳐지지 않고,
# 나름의 역할 수행이 가능함

# 시그모이드 활성화 함수를 사용한 은닉층과 소프트맥스 함수를 사용한 출력층을 Keras의 Dense 클래스로 구성
dense1 = keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784, ))
# dense1이 은닉층이고, 100개의 뉴런을 가진 밀집층의 활성화 함수를 시그모이드로 설정
# 뉴런 구성은 경험적으로 구성하며, 제약사항으로는 적어도 출력층의 뉴런보다는 높은 수로 구성
dense2 = keras.layers.Dense(10, activation = 'softmax')
# dense2는 출력층으로 설정하며, 활성화 함수는 소프트맥스 함수로 지정

# 심층 신경망 구성
model = keras.Sequential([dense1, dense2])
# 심층신경망 구성 시 입력층, 출력층 순서대로 구성
# 인공신경망의 강력한 성능은 층을 추가하여 입력데이터에 대해 연속적인 학습이 가능하다는 것임
print("layer 정보 출력")
model.summary()
print("=" * 50)
# 첫째줄 : model name
# 두번째줄 : model의 층 정보가 순서대로 출력(은닉충 -> 출력층 순)
    # 1열 : 층 이름(클래스) / 2열 : 출력크기 / 3열 : 모델 파라미터
# 층 이름 : 지정하지 않으면 자동으로 Dense로 지정
# 출력크기 : None(샘플의 개수 / 미지정), 뉴런 개수
    # fit() 매서드에서 batch_size 매개변수로 변경 가능
    # 샘플개수는 고정하지 않고, 유연하게 대응할 수 있도록 None으로 설정

# 배치차원 : 신경망층에서 입력되거나 출력되는 배열의 첫번째 차원
# 파라미터 : 파라미터 개수 출력 / {(입력데이터 개수 * 해당 layer 뉴런) + 해당 layer 뉴런}
#  Non-trainable params: 0 (0.00 B) : 훈련되지 않은 파라미터 값
    # -> 간혹 경사 하강법으로 훈련되지 않는 파라미터는 가진 층의 파라미터

# 심층신경망 구성의 두번째 방법

# Sequential 클래스에 층을 추가하는 다른방법
# Sequential 클래스 생성자 안에서 Dense 클래스의 객체를 만듦

model = keras.Sequential([
    keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784, ), name = 'hidden'),
    keras.layers.Dense(10, activation = 'softmax', name = 'output')],
    name = 'Fashion_MNIST_Model')    # model 네임

print("심층신경망 ver.2")
model.summary()
print("=" * 50)

# .add() 매서드 이용 layer 추가
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation = 'sigmoid', input_shape = (784, )))
model.add(keras.layers.Dense(10, activation = 'softmax'))
print("add() 적용 layer 추가")
model.summary()
print("=" * 50)

# 모델 훈련
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("모델 훈련")
model.fit(train_scaled, train_target, epochs = 5)
print("=" * 50)
print("모델 검증")
model.evaluate(val_scaled, val_target)
print("=" * 50)

# ReLU(렐루)함수
# 기존에는 시그모이드함수를 사용하였지만,
# 층이 많은 심층신경망 일수록 올마른 출력을 만드는데 신속하게 대응불가함
# 렐루함수는 입력이 양수일 경우 활성화함수가 없는거처럼 입력값을 그냥 통과하고, 음수일경우 0으로 만듦
# 특히, 이미지 처리에서 좋은 성능을 발휘함
# Flatten을 이용 배치 차원을 제외하고 나머지 입력차원을 모두 일렬로 펼침
    # Flatten은 입력층 바로 뒤에 사용

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add((keras.layers.Dense(10, activation = 'softmax')))
print("ReLU 함수 적용")
model.summary()
print("=" * 50)

# 2차원 해상도값 그대로 적용
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2)

model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print("2차원 배열 적용")
model.fit(train_scaled, train_target, epochs = 5)
print("=" * 50)

print("2차원 배열 검증")
model.evaluate(val_scaled, val_target)
print("=" * 50)

