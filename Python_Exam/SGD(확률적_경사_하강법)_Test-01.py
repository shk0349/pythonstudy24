import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

fish = pd.read_csv('https://bit.ly/fish_csv_data')
print("데이터의 상위 5개 출력")
print(fish.head())
print("=" * 50)

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

np.set_printoptions(precision = 6, suppress = True)    # 배열의 표기방법 변경
# precision = 6 : 배열 출력 시 소숫점 6자리까지 표시(Default = 8)
# suppress = True : 과학적 표기법(ex 1.23e+05) 대신 일반적인 소수점 표기법 사용

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)    # train_input 값 표준점수화
test_scaled = ss.transform(test_input)    # test_input 값 표준점수화
print('train_scaled')
print(train_scaled[:5])
print("=" * 50)
print('test_scaled')
print(test_scaled[:5])
print("=" * 50)


sc = SGDClassifier(loss = 'log_loss', max_iter = 10, random_state = 42)
# loss = 'log_loss' : 로지스틱 손실함수 지정
# max_iter = 10 : 10회 epoch(에포크) 실행
sc.fit(train_scaled, train_target)

print("SGD를 적용한 훈련값 / 에포크 10")
print(sc.score(train_scaled, train_target))
print("=" * 50)
print("SGD를 적용한 테스트값 / 에포크 10")
print(sc.score(test_scaled, test_target))
print("=" * 50)
# 결론 : 훈련값(77.3%) < 테스트값(77.5%)로 과소적합 -> iter_max 값을 높혀 학습횟수를 높이면 해결가능

# 훈련할 모델 추가
sc.partial_fit(train_scaled, train_target)
print("SGD를 적용한 훈련값 / 에포크 10 / 훈련모델 추가 및 적용")
print(sc.score(train_scaled, train_target))
print("=" * 50)
print("SGD를 적용한 테스트값 / 에포크 10 / 훈련모델 추가 및 적용")
print(sc.score(test_scaled, test_target))
print("=" * 50)
# 결론 : 점진적 학습을 통하여 첫 계산보다 정확도가 증가하나, 부분적인 학습을 계속 진행하는 것은 의미가 없으며, 기준점을 제공해야함 / 정체된 느낌을 줌
# 확률적 경사 하강법을 사용한 모델은 에포크 횟수에 따라 과소 또는 과대적합이 될 수 있음
# 에포크 횟수가 적으면 훈련세트를 덜 학습하여 과소적합, 에포크 횟수가 지나치게 많으면 과대적합이 될 확률이 높음
# 이를 방지하고자 과대적합이 되기전에 훈련을 종료하는 것을 조기종료라 함

sc = SGDClassifier(loss = 'log_loss', random_state = 42)
train_score = []
test_score = []
classes = np.unique(train_target)    # train_target의 목록 제공

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(sc.score(train_scaled, train_target))    # 빈 리스트에 훈련세트 점수 추가
    test_score.append(sc.score(test_scaled, test_target))    # 빈 리스트에 테스트세트 점수 추가

# 위 반복된 값을 그래프화
plt.plot(train_score)
plt.plot(test_score)
plt.title('SGD')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

sc = SGDClassifier(loss = 'log_loss', max_iter = 100, tol = None, random_state = 42)    # tol = None : 조기종료 안함
sc.fit(train_scaled, train_target)
print("SGD를 적용한 훈련값 / 에포크 100")
print(sc.score(train_scaled, train_target))
print("=" * 50)
print("SGD를 적용한 테스트값 / 에포크 100")
print(sc.score(test_scaled, test_target))
print("=" * 50)

# 손실함수에 대한 loss 매개값
# loss 매개값의 Default값 : loss = 'hinge'(힌지손실) -> 서포트 백터 머신(Support Vector Machine) : 또다른 머신러닝 알고리즘을 위한 손실함수
sc = SGDClassifier(loss = 'hinge', max_iter = 100, tol = None, random_state = 42)
sc.fit(train_scaled, train_target)
print("SGD를 적용한 훈련값 / 에포크 100 / hinge")
print(sc.score(train_scaled, train_target))
print("=" * 50)
print("SGD를 적용한 테스트값 / 에포크 100 / hinge")
print(sc.score(test_scaled, test_target))
print("=" * 50)