import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

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

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# 이웃의 갯수를 3으로 설정
knr3 = KNeighborsRegressor(n_neighbors = 3)
knr3.fit(train_input, train_target)
print("길이가 50cm인 농어의 무게 예측")
print(knr3.predict([[50]]))    # 길이가 50cm인 농어의 무게 예측
print("=" * 30)

distances, indexes = knr3.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.title('Length 50cm Perch weight predict - KNeighbors')
plt.xlabel('Length(cm)')
plt.ylabel('Weight(g)')
plt.show()
# 결과 : 기본데이터 부족으로 예측에 한계가 생김


# LinearRegression을 이용한 선형회귀
# 공식 / y(농어의 무게) = a(기울기) * x(농어의 길이) + b(절편 or 가중치)
# a = lr.coef+ / b = lr.intercept_
lr = LinearRegression()
lr.fit(train_input, train_target)
print("선형회귀를 이용한 길이 50cm 농어 무게 예측")
print(lr.predict([[50]]))
print("=" * 30)

plt.scatter(train_input, train_target)
# 15 ~ 50까지 1차방정식
plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])
plt.scatter(50, 1241.8, marker = '^')
plt.title('Length 50cm Perch weight predict - Linear')
plt.xlabel('Length(cm)')
plt.ylabel('Weight(g)')
plt.show()
# 결과 : 예측은 가능하나 기울기 값이 직선이라 결과값이 너무 직선적으로 나옴


# LinearRegression을 이용한 다항회귀(Polynomial Regression)
# 공식 / y(농어 무게) = {a1(기울기 시작지점 값) * x(농어 길이)^2} + {a2(기울기 끝지점 값 * x(농어 길이)} + b(절편 or 가중치)
# a1, a2는 lr.coef_에서 각각 첫번째, 두번째 값
# b는 lr.intercept_ 값
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr.fit(train_poly, train_target)
print("다항회귀를 이용한 길이 50cm 농어 무게 예측")
print(lr.predict([[50**2, 50]]))
print("=" * 30)

point = np.arange(15, 50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01 * point**2 + -21.6 * point + 116.05)    # 다항회귀 공식 적용
plt.scatter([50], [1574], marker = '^')
plt.title('Length 50cm Perch weight predict - Polynomial')
plt.xlabel('Length(cm)')
plt.ylabel('Weight(g)')
plt.show()