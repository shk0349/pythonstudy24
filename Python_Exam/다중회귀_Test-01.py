import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

df = pd.read_csv('http://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42)

poly = PolynomialFeatures(include_bias = False)    # bias(1) 값을 제외
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

lr = LinearRegression()
lr.fit(train_poly, train_target)
print("물고기의 길이, 높이, 두께까지 사용한 결과 / 트레이닝값")
print(lr.score(train_poly, train_target))
print("=" * 25)
print("물고기의 길이, 높이, 두께까지 사용한 결과 / 테스트값")
print(lr.score(test_poly, test_target))
print("=" * 25)

# 특성 조합 수를 5로 늘려 진행
poly = PolynomialFeatures(degree = 5, include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
lr.fit(train_poly, train_target)
print("물고기의 길이, 높이, 두께까지 사용한 결과 / 트레이닝값")
print(lr.score(train_poly, train_target))
print("=" * 25)
print("물고기의 길이, 높이, 두께까지 사용한 결과 / 테스트값")
print(lr.score(test_poly, test_target))    # 자료보다 특성이 많아 너무 과대적합한 결과값이 출력됨
print("=" * 25)

# 규제(Regularization : 훈련세트를 너무 과도하게 학습하여 과대적합이 되지 않도록 제한
ss = StandardScaler()
ss.fit(train_poly)
# 표준점수로 변환한 scale값 준비
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# Ridge Model / Default
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("Ridge Model을 이용한 결과 / 트레이닝값")
print(ridge.score(train_scaled, train_target))
print("=" * 25)
print("Ridge Model을 이용한 결과 / 테스트값")
print(ridge.score(train_scaled, train_target))
print("=" * 25)

# Ridge Model / Regularization
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    ridge = Ridge(alpha = alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
print("Ridge Model에 규제를 건 결과 / 트레이닝값")
print(train_score)
print("=" * 25)
print("Ridge Model에 규제를 건 결과 / 테스트값")
print(test_score)
print("=" * 25)

# 그래프로 표현
plt.plot(np.log10(alpha_list), train_score)    # x축 값이 너무 촘촘하니 지수형식으로 표현
plt.plot(np.log10(alpha_list), test_score)    # 0.001 -> -3 / 0.01 -> -2 / 0.1 -> -1 / 1 -> 0 / 10 -> 1 / 100 -> 2
plt.title('Ridge Model Regularization')
plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.show()

# 규제값 중 가장 좋은 값 선정 후 적용
ridge = Ridge(alpha = 0.1)
ridge.fit(train_scaled, train_target)

print("Ridge Model에 최적의 규제값을 적용한 결과 / 트레이닝값")
print(ridge.score(train_scaled, train_target))
print("=" * 25)
print("Ridge Model에 최적의 규제값을 적용한 결과 / 테스트값")
print(ridge.score(test_scaled, test_target))
print("=" * 25)

# Lasso Model / Default
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print("Lasso Model을 적용한 결과 / 트레이닝값")
print(lasso.score(train_scaled, train_target))
print("=" * 25)
print("Lasso Model을 적용한 결과 / 테스트값")
print(lasso.score(test_scaled, test_target))
print("=" * 25)

# Lasso Model / Regularization
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # max_iter : 최대반복횟수(정확도 향상을 위함)
    # Lasso Model은 최적의 계수를 찾기 위해 반복 계산 실시
    lasso = Lasso(alpha = alpha, max_iter = 10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
print("Lasso Model에 규제를 건 결과 / 트레이닝값")
print(train_score)
print("=" * 25)
print("Lasso Model에 규제를 건 결과 / 테스트값")
print(test_score)
print("=" * 25)
# Lasso Model은 Test 시 반복적으로 계산을 수행하나, 지정한 반복횟수가 부족하면 경고가 발생함(ConvergenceWarning)

# 그래프로 표현
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.title('Lasso Model Regularization')
plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.show()

# 규제값 중 가장 좋은 값을 선정후 적용
lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print("Lasso Model에 최적의 규제값을 적용한 결과 / 트레이닝값")
print(lasso.score(train_scaled, train_target))
print("=" * 25)
print("Lasso Model에 최적의 규제값을 적용한 결과 / 테스트값")
print(lasso.score(test_scaled, test_target))
print("=" * 25)