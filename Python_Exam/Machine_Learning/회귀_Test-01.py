import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

knr = KNeighborsRegressor()

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
                        21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
                        23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
                        27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
                        39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5, 44.0])

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
                        115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
                        150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
                        218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
                        556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
                        850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0, 1000.0])

plt.rc('font', family='Malgun Gothic')
plt.scatter(perch_length, perch_weight)
plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('농어 길이와 무게의 상관관계')
plt.show()

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

knr.fit(train_input, train_target)
print('테스트 세트에 있는 샘플을 정확하게 분류한 갯수')
print(knr.score(test_input, test_target))
print("="*30)

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)

print('타깃과 예측값 사이의 차이')
print(mae)
print("="*30)

print('훈련한 모델을 사용한 훈련세트 점수')
print(knr.score(train_input, train_target))
print("="*30)

# 사이킷 런의 알고리즘 기본값을 3개로 조절하여 적용
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print('재설정된 값으로 모델 재훈련')
print(knr.score(train_input, train_target))
print("="*30)

print('기본값 조정으로 과대/과소적합 해결')
print(knr.score(test_input, test_target))
print("="*30)

x = np.arange(5, 45).reshape(-1, 1)

for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    knr.n_neighbors = n
    knr.fit(train_input, train_target)
    prediction = knr.predict(x)
    plt.scatter(train_input, train_target)
    plt.plot(x, prediction)
    plt.title('n_neighbors = {}'.format(n))
    plt.xlabel('Length(cm)')
    plt.ylabel('Weight(g)')
    plt.show()