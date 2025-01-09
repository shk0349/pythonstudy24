import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

wine = pd.read_csv('https://bit.ly/wine_csv_data')
print("데이터 정보")
print(wine.info())
print("=" * 50)
print("데이터 통계")
print(wine.describe())
print("=" * 50)

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# test_size를 사용하여 훈련세트(80%)와 테스트세트(20%)로 분할
train_input, test_input, train_target, test_target = train_test_split(data, target,test_size = 0.2 , random_state = 42)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print("LogisticRegression 적용 훈련세트")
print(lr.score(train_scaled, train_target))
print("=" * 50)
print("LogisticRegression 적용 테스트세트")
print(lr.score(test_scaled, test_target))
print("=" * 50)
# 결론 : 훈련세트(78.08%) > 테스트세트(77.77%)이나 둘다 80% 미만의 값이므로 과소적합

# 로지스틱 회귀의 계수 및 절편
print(lr.coef_, lr.intercept_)
print("=" * 50)

# Decision_Tree(결정트리) / if문을 사용하는것과 유사
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)

print("DecisionTree 적용 훈련세트")
print(dt.score(train_scaled, train_target))
print("=" * 50)
print("DecisionTree 적용 테스트세트")
print(dt.score(test_scaled, test_target))
print("=" * 50)
# 결론 : 훈련세트(99.7%) > 테스트세트(85.9%)이나 훈련세트의 값이 지나치게 높아 과대적합

# 그래프 작성 - 1
plt.figure(figsize = (10, 7))    # 가로 10인치, 세로 7인치 사이즈 그래프
plot_tree(dt)
plt.show()    # 최상위 노드 = 루트노드(root_nod) / 맨 아래노드 = 리프노드(leaf_nod)

# 그래프 작성 - 2
plt.figure(figsize = (10, 7))
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
# max_depth = 1 : 루트노드(root_nod) 제외
# filled = True : 클래스에 맞게 색 적용
# feature_names = ['alcohol', 'sugar', 'pH']
plt.show()
# sugar <= -0.239 : True / sugar >= -0.239 : False
# 총 샘플수 : 5,197개 / 음성클래스(레드와인) : 1,258개 / 양성클래스(화이트와인) : 3,939개
    # 왼쪽 노드는 항상 당도가 더 낮은지를 물어보는 노드 / 오른쪽 노드는 그 반대
    # 두번째 노드에서 왼쪽은 음성클래스 1,177개, 양성클래스 1,745개로 구성
    # 오른쪽 노드는 음성클래스 81개, 양성클래스 2,194개로 구성
# gini = 0.367 : 불순도(노드에서 데이터를 분할할 기준을 정하는 값)
# samples = 5197 : 총 샘플수
# value = [1258, 3939] : 값(1 : 1258개, 0 : 3939개)
# 노드의 바탕색 중 오른쪽 노드는 루트노드(맨 상단노드)보다 더 진하고 왼쪽노드는 연해짐
# filled = True : 클래스마다 색을 지정하며, 비율이 높아지면 색이 점점 진하게 표시됨
# 결정트리에서 예측하는 방법 : 리프노드에서 가장 많은 클래스가 예측 클래스가 되며,
# 만약 이 결정 트리로 성장을 멈춘다면 왼쪽 노드에 도달한 샘플과 오른쪽 노드에 도달한 샘플은 모두 양성 클래스로 예측됨(양성클래스가 많기 때문)

# 가지치기 : 열매를 잘 맺기 위하여 가지치기를 하듯이 결정트리에서도 사용함
# 가지치기를 하지 않으면 무작정 끝까지 자라나는 트리를 만듬 -> 트리의 길이(depth)를 지정
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_scaled, train_target)

print("가지치기 적용 훈련세트")
print(dt.score(train_scaled, train_target))
print("=" * 50)
print("가지치기 적용 테스트세트")
print(dt.score(test_scaled, test_target))
print("=" * 50)

# 그래프 작성 - 3
plt.figure(figsize = (20, 15))
plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()
# 그래프 분석
# 루트노드(depth = 0) 다음 깊이에 있는 depth = 1 노드는 모두 당도를 기준으로 훈련세트를 나눔
# 하지만 depth = 2에서는 맨 왼쪽 노드만 당도를 기준으로 나누고 왼쪽에서 2번째 노드는 알콜도수기준으로 나눔
# 오른쪽 두 노드는 pH를 기준으로 나눔
# depth = 3에 있는 노드가 최종 노드(리프노드)
# 왼쪽에서 3번째에 있는 노드만 음성클래스가 더 많음 / 이 노드에 도착해야만 레드와인으로 예측됨
# 결론 : -0.239 < sugar < -0.802 / alcohol < 0.454  ->  2개 경우의 수를 모두 충족하면 레드와인 / 나머지는 화이트와인

# -0.802라는 음수로 된 당도는 보고용으로 사용불가
# 표준화 전처리를 하지 않고 결정트리 사용(특성값의 스케일화는 효과가 없음)
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_input, train_target)

print("표준화 전처리 미적용 훈련세트")
print(dt.score(train_scaled, train_target))
print("=" * 50)
print("표준화 전처리 미적용 테스트세트")
print(dt.score(test_scaled, test_target))
print("=" * 50)

# 그래프 작성 - 4
plt.figure(figsize = (20, 15))
plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()
# 1.625 < sugar < 4.325 / alcohol < 11.025 -> 두 조건을 모두 만족하면 레드와인으로 판단

# 결정트리는 어떠한 특성이 가장 유용하였는지를 나타대는 특성 중요도를 계산해줌
# .feature_importances의 인수들의 합이 1이 되어야함
# alcohol, sugar, pH 순으로 표기되어있음(입력값의 순서대로 표기)
print("특성 중요도")
print(dt.feature_importances_)
print("=" * 50)

# 좌우 불균형 트리
dt = DecisionTreeClassifier(min_impurity_decrease = 0.0005, random_state = 42)    # min_impurity_decrease : 최소 불순도
dt.fit(train_input, train_target)
print("좌우 불균형 트리형태 적용 훈련세트")
print(dt.score(train_scaled, train_target))
print("=" * 50)
print("좌우 불균형 트리형태 적용 테스트세트")
print(dt.score(test_scaled, test_target))
print("=" * 50)

# 그래프 작성 - 5
plt.figure(figsize = (20, 15))
plot_tree(dt, filled = True, feature_names = ['alcohol', 'sugar', 'pH'])
plt.show()