import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import uniform, randint

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련세트(80%)와 테스트세트(20%)로 분할
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2)
# 훈련세트를 서브세트(80%)와 검증세트(20%)로 분할
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size = 0.2)

dt = DecisionTreeClassifier()
dt.fit(sub_input, sub_target)

print("DecisionTree 적용 서브세트 결과")
print(dt.score(sub_input, sub_target))
print("=" * 50)
print("DecisionTree 적용 검증세트 결과")
print(dt.score(val_input, val_target))
print("=" * 50)

# 검증세트 생성을 위해 훈련세트 감소(많은 데이터 이용 훈련 시 좋은 결과 도출 가능)
# 교차검증 : 검증 세트를 떼어내어 평가하는 과정 여러번 반복
# 패리티 방식(3-폴드 교차 검증 / 5-폴드, 10-폴드도 있음) -> sklearn의 cross_validate()라는 교차검증 함수 이용
# 예시
# [훈련세트, 훈련세트, 검증세트]
# [훈련세트, 검증세트, 훈련세트]
# [검증세트, 훈련세트, 훈련세트]

# 평가할 모델 객체를 첫번째 매개변수로 전달 / 직접 검증세트를 분할하지 않고 훈련세트 전체 전달
scores = cross_validate(dt, train_input, train_target)    # 매개변수 cv를 이용하여 폴드 수 변경가능하며, devault값 5임

print("교차검증을 활용하여 DecisionTree 적용 훈련세트 결과")
print(scores)
print("=" * 50)
print("교차검증을 활용하여 DecisionTree 적용 훈련세트 검증점수 평균값")
print(np.mean(scores['test_score']))
print("=" * 50)
# 주의사항 : cross_validate()는 훈련세트를 섞어서 폴드 분할하지 않음 -> 분할기(splitter) 사용
# train_test_split() 함수로 전체 데이터를 섞은 후 훈련세트 준비하였으나, 교차검증 시 훈련세트를 섞으려면 분할기(splitter) 지정
# sklearn의 분할기는 교차검증에서 폴드를 어떻게 나눌지 결정
# cross_validate()는 기본적으로 회귀모델인 KFold 분할기 사용
# 분류 모델일 경우 target class를 골고루 나누기 위해여 StratifiedKFold 사용

scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
print("교차검증을 활용하여 DecisionTree 적용 훈련세트 검증점수 평균값 / cv값 = StratifiedKFold 적용")
print(np.mean(scores['test_score']))
print("=" * 50)

# 결정트리의 매개변수 값을 바꿔가며 가장 좋은 성능이 나오는 모델을 찾아야함
# 테스트세트를 사용하지 않고 교차검증을 통해 좋은 모델 선택

# 하이퍼파라미터 튜닝
# 모델 파라미터 : Machine Learning Model이 학습하는 Parameter
# 하이퍼파라미터 : 사용자가 지정해야만 하는 Parameter
# 하이퍼파라미터 튜닝 : Library가 제공하는 Default값을 그대로 사용하여 Model 훈련
    # 이후 검증세트의 점수나 교차검증을 통하여 매개변수를 조금씩 변경하여 1~2, 5~6개의 매개변수 제공
# AutoML : 사람의 개입없이 하이퍼파라미터 튜닝을 자동으로 수행하는 기술

# Max_depth를 최적으로 고정하고 min_sample_split을 바꿔가며 최적의 값을 찾음 -> 면 값이 함께 변경
# -> 두개의 매개변수를 동시에 바꿔가며 최적의 값을 찾아야함 -> 복잡해짐 -> sklearn의 GridSearchCV

# min_impurity_decrease 매개변수의 최적값 확인
# min_sample_split : 샘플을 최소한 몇개 이상이어야 split(하위노드로 분리) / 과대적합 방지를 위해 클수록 가지치기 시행, 작을수록 정확함
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
# 매개변수와 탐색할 값의 리스트를 딕셔너리로 생성

gs = GridSearchCV(DecisionTreeClassifier(), params, n_jobs = -1)
# 결정트리 클래스의 객체를 생성하고 바로 전달
# 일반 모델을 훈련하는 것처럼 .fit() 매서드 호출 -> 서치객체는 결정트리모델 min_impurity_decrease 값을 변경하면서 총 5번 수행
# GridSearchCV의 매개변수 cv의 Default값 = 5 / 5 * 5 = 25회 수행
# n_jobs : 병렬시행에 사용될 CPU 코어수 지정(-1 : 모든 코어 사용)

gs.fit(train_input, train_target)
# 교차검증에서 최적의 하이퍼파라미터를 찾으면 전체 훈련세트로 모델을 생성해야하나,
# sklearn의 GridSearchCV는 검증 점수가 가장 높은 모델의 매개변수 조합으로
# 전체 훈련세트에서 자동으로 다시 모델을 훈련함

# 이와 같이 최적화되어 훈련된 모델은 GridSearchCV 클래스로 만들어진 객체의 best_estimator_ 속성에,
# 최적의 매개변수는 best_params_ 속성에 저장되어 있음
print(f"그리드 서치를 통해 찾은 최적의 모델 정확도 : {dt.score(train_input, train_target) * 100 : .2f}%")
print("=" * 50)
print(f"최적의 매개변수 : {gs.best_params_}")
print("=" * 50)

print("5번의 교차검증으로 얻은 점수")
print(gs.cv_results_['mean_test_score'])
print("=" * 50)

# 눈으로 보는 것 보다 numpy의 argmax() 함수를 사용하면 가장 큰 값의 인덱스 추출 가능
# 해당 인덱스를 이용하여 params 키값에 저장된 매개변수 출력
# 이 값이 최상의 검증 점수를 만든 매개변수 조합
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print("최상의 검증점수 출력")
print(gs.cv_results_['params'][best_index])
print("=" * 50)
# 과정요약
# 1. 탐색할 매개변수 지정
# 2. 훈련세트에서 그리드 서치를 수행하여 최상의 평균점수가 나오는 매개변수 조함들 찾아 그리드 서치에 저장
# 3. 그리드 서치는 최상의 매개변수에서 전체 훈련세트를 사용하여 최종모델 훈련하여 그리드 서치에 저장

# 더 복잡한 매개변수 조합 시행
# 노드를 분할하기 위한 불순도 감소 최소량 지정 / max_depth(트리의 깊이)
# min_samples_split : 노드를 나누기 위한 최소 샘플 수
# numpy의 arange() 함수는 첫번째 매개변수 값에서 시작하여
# 두번째 매개변수에 도달할 때 까지 세번째 매개변수를 계속 더한 배열 생성
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
          'max_depth' : range(5, 20, 1),
          'min_samples_split' : range(2, 100, 10)}
# 총 교차 검증 횟수 : 1350(9 * 15 * 10) * 5(5-폴드교차) = 6750회

# criterion : 분할 품질을 측정하는 기능 (default : gini)
# splitter : 각 노드에서 분할을 선택하는데 사용되는 전략 (default : best)
# max_depth : 트리의 최대 깊이(값이 클수록 모델의 복잡도 상승)
# min_samples_split : 자식 노드를 분할하는데 필요한 최소 샘플의 수 (default : 2)
# min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수 (default : 1)
# min_weight_fraction_leaf : min_samples_leaf와 같지만 가중치가 부여된 샘플 수에서의 비율
# max_features : 각 노드에서 분할에 사용할 특징의 최대 수
# random_state : 난수 seed 설정
# max_leaf_nodes : 리프 노드의 최대 수
# min_impurity_decrease : 최소 불순도
# min_impurity_split : 나무 성장을 멈추기 위한 임계치
# class_weight : 클래스 가중치
# presort : 데이터 정렬 필요 여부

gs = GridSearchCV(DecisionTreeClassifier(random_state = 42), params, n_jobs = -1)
gs.fit(train_input, train_target)

print("최상의 매개변수 조합")
print(gs.best_params_)
print("=" * 50)
# 탐색할 매개변수 간격을 0.0001이나 1로 설정하였는데, 이는 근거가 부족함(좁히거나 넓혀야할 필요성 상존)

# 매개변수의 값이 수치로 나타날때 값의 범위나 간격을 미리 정하기 어려움
# 너무 많은 매개변수 조건이 있어 그리드 서치 수행시간이 오래걸림 -> 랜덤 서치 사용하여 해결 가능

# 랜덤서치 : 매개변수 값의 목록을 전달하는 것이 아니라 매개변수의 샘플링을 할 수 있는 확률 분포도 객체 전달
# 싸이파이 : Python의 핵심 과학 Library로 적분, 보간, 선형대수, 확률 등을 포함한 수치 계산용 전용 Library

# uniform, randint 클래스는 모두 주어진 범위에서 고르게 값을 뽑음(균등 분포에서 샘플링)
# randint()에서는 정수값을 추출, uniform()은 실수값 출력

params = {'min_impurity_decrease' : uniform(0.0001, 0.001),    #  0.0001 ~ 0.001 사이의 실수값
          'max_depth' : randint(20, 50),    # 20 ~ 50 사이의 정수
          'min_samples_split' : randint(2, 25),    # 2 ~ 25 사이의 정수
          'min_samples_leaf' : randint(1, 25)}    # 1 ~ 25 사이의 정수
# 리프노드가 되기 위한 최소 샘플 개수(자식노드의 샘플 수보다 이 값이 작으면 분할불가)

# 샘플링 횟수는 sklearn의 RandomizedSearchCV의 n_iter 매개변수에 지정
gs = RandomizedSearchCV(DecisionTreeClassifier(), params, n_iter = 100, n_jobs = -1)
# n_iter = 100 : 총 100번 샘플링하여 교차검증 수행(최적의 매개변수 조합을 찾음)
gs.fit(train_input, train_target)

print("최적의 매개변수 조합")
print(gs.best_params_)
print("=" * 50)
print("최고의 교차 검증 점수")
print(np.max(gs.cv_results_['mean_test_score']))
print("=" * 50)

# 최종모델로 결정 및 테스트세트 성능 확인
dt = gs.best_estimator_
print("최종모델 적용 테스트세트")
print(dt.score(test_input, test_target))
print("=" * 50)
# 결과 : 검증세트(86.95%) > 테스트세트(86.0%)로 과대 및 과소적합에 미해당되며, 적당한 값으로 출력됨