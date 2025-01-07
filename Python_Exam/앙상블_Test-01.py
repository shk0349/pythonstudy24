import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 앙상블 : 여러 단순한 모델을 결합하여 정확한 모델을 만드는 방법
# 정형데이터 : 수치자료가 있는 값
# 비정형데이터 : 데이터베이스나 엑셀로 표현하기 어려운 데이터(textdata, image, audio 등) -> 신경망 알고리즘 사용
# RandomForest : DecisionTress를 랜덤하게 만들어 DecisionTree 숲을 만듦 -> 최종예측
# Boodstrap : 데이터세트에서 중복을 허용하여 데이터 샘플링

# RandomForestClassifier : 기본적으로 전체 특성 개수의 제곱근만큼 특성 선택
    # 예시) 4개의 특성이 있다면 노드마다 2개를 랜덤하게 선택하여 사용
    # 단, 회귀모델인 RandomForestRegressor는 전체 특성 사용

# sklearn의 RandomForest는 기본적으로 100개의 DecisionTree를 아래 방식으로 훈련함
    # 분류 : 각 트리의 클래스별 확률의 평균을 구하여 가장 높은 확률을 가진 클래스로 예측
    # 회귀 : 단순히 각 트리 예측의 평균을 구함
# 분류 : 샘플을 몇 개의 클래스 중 하나로 분류하는 것
# 회귀 : 임의의 숫자를 예측하는 것

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2)

rf = RandomForestClassifier(n_jobs = -1)    # n_jobs = -1 : 모든 cpu 사용
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs = -1)
# return_train_score = True : 검증점수와 훈련세트에 대한 점수 리턴

print("RandomForest 적용 훈련 및 테스트세트 평균")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print("=" * 50)

# RandomForest는 DecisionTree의 앙상블이기 때문에 DecisionTree가 제공하는 중요한 매개변수를 모두 제공
# DecisionTree의 큰 장점중 하나가 특성중요도 계산 가능
# RandomForest의 특성중요도는 각 DecisionTree의 특성중요도를 취합한 값

rf.fit(train_input, train_target)
print("RandomForest 특성중요도 출력")
print(rf.feature_importances_)
print("=" * 50)
# RandomForest가 특성 일부를 랜덤하게 선택하여 DecisionTree를 훈련함(중복허용)
# 하나의 특성에 과도하게 집중하지 않고, 여러 특성이 폭넓게 훈련에 기여가능(과대적합을 줄임)

# 자체적으로 모델을 평가하는 점수를 얻을 수 있음
# OOB(Out Of Bag) : Bootstrap에 포함되지 않고 남은 샘플 -> DecisionTree 평가용(검증세트로 활용)
# oob_score = True : RandomForest는 각 DecisionTree의 oob 점수를 평균하여 출력

rf = RandomForestClassifier(oob_score = True, n_jobs = -1)
rf.fit(train_input, train_target)
print("RandomForest oob score 출력")
print(rf.oob_score_)
print("=" * 50)

# ExtraTree : 100개의 DecisionTree 훈련 -> Bootstrap 샘플을 사용하지 않고 전체 훈련세트 사용
    # 대신 노드 분할 시 가장 좋은 분할을 찾는 것이 아니라 무작위로 분할
    # ExtraTree의 DedisionTree splitter(분할기) = 'random'
# 하나의 DecisionTree에서 특성을 무작위로 분할한다면 성능이 낮아지지만,
# 많은 트리를 앙상블하기 때문에 과대적합을 막고 검증세트의 점수를 높이는 효과가 있음

et = ExtraTreesClassifier(n_jobs = -1)
scores = cross_validate(et, train_input, train_target, return_train_score = True, n_jobs = -1)

print("ExtraForest 적용 훈련 및 테스트세트 평균")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print("=" * 50)
et.fit(train_input, train_target)
print("ExtraForest 특성중요도 출력")
print(et.feature_importances_)
print("=" * 50)

# 그레디언트 부스팅(Gradient Boosting) : 기울기 / 깊이가 얕은 결정트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블
# sklearn의 GradientBoostingClassifier는 기본적으로 깊이가 3인 결정트리 100개 사용
# 때문에 과대적합에 강하며, 높은 일반화 성능을 기대함
# Gradient는 경사 하강법을 사용하여 트리를 앙상블에 추가함
# 분류 : 로지스틱 손실함수, 회귀에서는 평균 제곱 오차 함수를 사용
# 경사 하강법 손실함수를 산으로 정의하고, 가장 낮은 곳으로 찾아 내려오는 과정

# 가장 낮은곳으로 내려오는 방법은 모델의 가중치와 절편을 조금씩 바꾸는것
# 결정 트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동

gb = GradientBoostingClassifier()
scores = cross_validate(gb, train_input, train_target, return_train_score = True, n_jobs = -1)

print("GradientBoosting 적용 훈련 및 테스트세트 평균")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print("=" * 50)
gb.fit(train_input, train_target)
print("GradientBoosting 특성중요도 출력")
print(gb.feature_importances_)
print("=" * 50)

# 히스토그램 기반 GradientBoosting(Histogram Gradient) : 그래디언트 부스팅의 개선버전
# 입력 특성을 256 구간으로 나눔 -> 노드를 분할할 때 최적의 분할을 매우 빠르게 찾을 수 있음
# 특히 256 구간 중에서 하나를 떼어놓고 누락된 값을 위해서 사용함
# HistoGradientBoostingClassifier는 기본 매개변수에서 안정적인 성능을 얻을 수 있음
# HistoGradientBoostingClassifier에는 트리의 개수를 지정하는데 n_estimators 대신 max_iter를 사용함(성능향상용)

hgb = HistGradientBoostingClassifier()
scores = cross_validate(hgb, train_input, train_target, return_train_score = True, n_jobs = -1)

print("Histogram 적용 훈련 및 테스트세트 평균")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print("=" * 50)

# permutation_importance : 특성 중요도 확인 / 특성을 하나씩 랜덤하게 섞어 모델의 성능이 변화되는지 관찰
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats = 10, n_jobs = -1)
print("Histogram 특성중요도 출력 / 훈련세트")
print(result.importances_mean)
print("=" * 50)

result = permutation_importance(hgb, test_input, test_target, n_repeats = 10, n_jobs = -1)
print("Histogram 특성중요도 출력 / 테스트세트")
print(result.importances_mean)
print("=" * 50)

# HistGradientBoostingRegressor 히스토그램 기반 그레디언트 부스팅의 회귀버전
# sklearn 말고도 그레이덩트 부스팅 알고리즘을 구현한 라이브러리가 다수 존재 -> XGBoost 대표적 -> cross_validate() 이용 교차검증 가능
xgb = XGBClassifier(tree_method = 'hist')    # tree_method = 'hist' : 히스토그램 기반 그레이디언트 부스팅
scores = cross_validate(xgb, train_input, train_target, return_train_score = True, n_jobs = -1)

print("XGBoost 적용 훈련 및 테스트세트 평균")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print("=" * 50)

# LGBMClassifier : ms에서 만든 LightGBM^2
lgb = LGBMClassifier()
scores = cross_validate(lgb, train_input, train_target, return_train_score = True, n_jobs = -1)

print("LGB 적용 훈련 및 테스트세트 평균")
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
print("=" * 50)