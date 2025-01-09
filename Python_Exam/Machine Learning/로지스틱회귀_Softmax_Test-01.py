import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.special import softmax

fish = pd.read_csv('https://bit.ly/fish_csv_data')
print("최상단 데이터 5개 출력")
print(fish.head())
print("=" * 80)

print("unique 매서드를 사용하여 생선 종류 출력")
print(pd.unique(fish['Species']))
print("=" * 80)

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

kn = KNeighborsClassifier(n_neighbors = 3)    # 참고 이웃 3개로 설정
kn.fit(train_scaled, train_target)
print("참고 이웃 3개로 설정된 결과 / 트레이닝 값")
print(kn.score(train_scaled, train_target))
print("=" * 80)
print("참고 이웃 3개로 설정된 결과 / 테스트 값")
print(kn.score(test_scaled, test_target))
print("=" * 80)

print("어종 출력")
print(kn.classes_)
print("=" * 80)

print("predict 메서드를 이용하여 test_target의 0 ~ 4 인덱스 값 확인")
print(kn.predict(test_scaled[:5]))
print("=" * 80)

proba = kn.predict_proba(test_scaled[:5])
print("test_target 0 ~ 4 인덱스의 확률")
print(np.round(proba, decimals = 4))
print("=" * 80)

distances, indexes = kn.kneighbors(test_scaled[3:4])
print("test_target 0 ~ 4 인덱스의 결과")
print(train_target[indexes])
print("=" * 80)

# LogisticRegression 알고리즘
# 공식 / z = (a * k1) + (b * k2) + (c * k3) + (d * k4) + ..... + (i * ki) + y
# a ~ i 는 각 항목에 대한 가중치(중요도, 계수), k1 ~ ki는 데이터 input 값
# z는 어떠한 값도 사용 가능하나, 확률이 되려면 0 ~ 1 사이 값이 되어야함
# z가 아주 큰 음수일때 0, 아주 큰 양수일 때 1이 되도록 시그모이드 함수, 로지스틱 함수을 사용하여 변환

z = np.arange(-5, 5, 0.1)    # -5 ~ 5까지 0.1간격으로 배열 생성
phi = 1 / (1 + np.exp(-z))    # 시그모이드 함수(exp 매서드) 적용 / z의 음수 값을 대입
plt.plot(z, phi)
plt.title('Sigmoid Function Example')
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# LogisticRegression / 2진분류
# Boolean Indexing : numpy 배열은 True, False 값을 전달하여 행을 선택할 수 있음
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print("Boolean Indexing Example")
print(char_arr[[True, False, True, False, False]])
print("=" * 80)

# Bream or Smelt 인 경우만 골라내어 설정
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')

train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print("train_bream_smelt의 0 ~ 4 인덱스의 값 출력")
print(lr.predict(train_bream_smelt[:5]))
print("=" * 80)

print("train_bream_smelt의 0 ~ 4 인덱스의 확률 출력")
print(lr.classes_)    # 클래스 확인
print(lr.predict_proba(train_bream_smelt[:5]))
print("=" * 80)

print("LogisticRegression으로 학습한 계수")
print(".coef_ / intercept_")
print(lr.coef_, lr.intercept_)
print("=" * 80)

decisions = lr.decision_function(train_bream_smelt[:5])
print("train_bream_smelt의 0 ~ 4 인덱스 값 출력")
print(decisions)
print("=" * 80)

# scipy 라이브러리 expit() 매서드 이용 시그모이드 함수에 z값을 대입하면 확률 도출 가능
print("scipy의 expit 매서드 이용 Sigmoid Function 사용")
print(expit(decisions))
print("=" * 80)

# Data 종류가 7가지이므로 True, False로만 처리하는 2진분류 대신에 LogisticRegression으로 변경하여 적용
# LogisticRegression 클래스 특징
# 1. 기본적으로 반복적인 알고리즘 사용
# 2. max_iter값이 부족하면 경고문구가 출력되니 값을 높여 적용(max_iter의 default값 : 100)
# 3. Ridge 회귀와 동일하게 계수를 제곱하는 규제 사용(L2 규제)
# 4. Ridge 회귀의 규제양이 alpha 매개변수 값과 비례한것과 반대로 LogisticRegression 규제양은 변수 C의 값과 반비례함

# C값을 20으로 높혀 규제완화 / 정확도 향상을 위하여 max_iter을 1000으로 증가
lr = LogisticRegression(C = 20, max_iter = 1000)
lr.fit(train_scaled, train_target)

print("LogisticRegression를 사용한 트레이닝값")
print(lr.score(train_scaled, train_target))
print("=" * 80)
print("LogisticRegression를 사용한 테스트값")
print(lr.score(test_scaled, test_target))
print("=" * 80)

print("test_scaled의 0 ~ 4 인덱스 결과를 다중분류 방식 출력")
print(lr.predict(test_scaled[:5]))
print("=" * 80)

proba = lr.predict_proba(test_scaled[:5])
print("target 종류의 알파벳 순 확률 출력")
print(lr.classes_)    # target 종류 출력
print(np.round(proba, decimals = 3))
print("=" * 80)

# 다중분류일때 선형방정식
# lr.coef_.shape : (i, j) 형식으로 출력되며, i개의 행과 j개의 특성이 사용됨
# lr.intercept_.shape : (k, ) 형식으로 출력되며, k개의 행이 사용됨 -> 2진분류에서 보았던 z값 k개를 계산함
# 다중분류는 클래스마다 z값을 한번씩 계산하며 가장 높은 z값을 출력하는 클래스가 예측값임
# 2진분류에서는 Sigmoid Function을 사용하여 z값을 0 ~ 1 사이값으로 변환하였으나,
# 다중분류에서는 Softmax Function을 사용하여 i개의 z값을 확률로 변환함
# Softmax 계산법
# e_sum = e^z1 + e^z2 + e^z3 + e^z4 + e^z5 + e^z6 + e^z7 + ..... + e^zi
# s1 = (e^z1 / e_sum).....s7 = (e^z7 / e_sum).....si = (e^zi / e_sum) -> s1 ~ s7까지 모두 더하면 분모가 같아져 1이 됨

decision = lr.decision_function(test_scaled[:5])
print("test_scaled의 0 ~ 4 인덱스값을 decision_function 함수에 넣은 값")
print(decision)
print("=" * 80)

print("5개 샘플에 대한 z1 ~ z7의 값\n")
for idx, z in enumerate(decision):
    print(f"{idx}번째 샘플의 z값\n{z}\n")

print("=" * 80)
print("Softmax Function을 이용한 예측확률 출력")
class_ = lr.classes_.tolist() + ["예측 결과"]    # target 리스트 + "예측결과" 출력
prd = lr.predict(test_scaled[:5]).reshape(5, -1)    # test_scaled의 0 ~ 4번 인덱스의 결과값을 5행으로 맞추고 열은 자동계산 설정 / 예측결과
sm = softmax(decision, axis = 1).round(2) * 100    # 각 샘플에서 softmax 연산 수행한 후 소숫점 2자리까지 반올림 및 100을 곱하여 확률로 표현 / 확률
con = np.column_stack((sm, prd))    # sm(확률)과 prd(예측결과) 두가지 배열을 결합하여 하나의 배열로 만듬
print(pd.DataFrame(con, columns = class_))    # target list를 첫 행에 배치하고 다음행부터는 con(확률 및 예측결과 배열) 출력