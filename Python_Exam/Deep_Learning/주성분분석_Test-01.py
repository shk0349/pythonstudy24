import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

# url에서 파일 다운로드
url = "https://bit.ly/fruits_300_data"
output_file = "fruits_300.npy"

try:
    urllib.request.urlretrieve(url, output_file)
    print(f"File download is successful: {output_file}")
except Exception as e:
    print(f"File download is failed: {e}")

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)

# pca 활용 주성분분석 알고리즘 사용
pca = PCA(n_components = 50)    # 주성분개수 50개 지정
pca.fit(fruits_2d)

def draw_fruits(arr, ratio = 1):    # ratio = 비율, 비
    n = len(arr)    # n : 샘플개수 / 한줄에 10개씩 이미지 그림
    rows = int(np.ceil(n / 10))    # 샘플 개수를 10으로 나누어 전체 행 개수를 계산
    cols = n if rows < 2 else 10    # 행이 1이면 열 개수는 샘플개수 / 그렇지 않으면 10개
    fig, axs = plt.subplots(rows, cols, figsize = (cols * ratio, rows * ratio), squeeze = False)
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:    # n개까지만 그림
                axs[i, j].imshow(arr[i * 10 + j], cmap = 'gray_r')
            axs[i, j].axis('off')
    plt.show()

draw_fruits(pca.components_.reshape(-1, 100, 100))

# transform() 매서드 활용 원본데이터 차원 감소
fruits_pca = pca.transform(fruits_2d)

# 10,000개 특성을 50개로 줄여 데이터에 손실이 발생하였으나,
# 최대한 분산이 큰 방향으로 데이터를 투영하였기 때문에,
# 원본데이터를 상당부분 재구성 가능
# inverse_tansform() 활용 재구성
fruits_inverse = pca.inverse_transform(fruits_pca)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")

print("이미지 품질")
print(np.sum(pca.explained_variance_ratio_))
print("=" * 50)

plt.plot(pca.explained_variance_ratio_)

# 위에 만든 주성분을 LogisticRegression에 적용
lr = LogisticRegression()
# 지도학습 모델을 사용하기 위해 타깃을 생성 / 사과 = 0, 파인애플 = 1, 바나나 = 2
target = np.array([0] * 100 + [1] * 100 + [2] * 100)

scores = cross_validate(lr, fruits_2d, target)
print("fruits_2d 원본데이터 확률")
print(np.mean(scores['test_score']))    # 99.67%의 과대적합 / 특성이 10,000개나 되기 때문에 300개의 샘플에서는 과대적합 모델이 됨
print("=" * 50)
print("fruits_2d 원본데이터 연산시간")
print(np.mean(scores['fit_time']))    # 1.13초 정도 걸림
print("=" * 50)

scores = cross_validate(lr, fruits_pca, target)    # fruits_pca PCA로 축소한 자료 사용
print("fruits_pca PCA로 축소한 자료 확률")
print(np.mean(scores['test_score']))    # 99.67%의 과대적합 / 원본데이터때와 동일
print("=" * 50)
print("fruits_pca PCA로 축소한 자료 연산시간")
print(np.mean(scores['fit_time']))    # 0.014초로 빠르게 처리
print("=" * 50)
# 결론 : 처리 속도가 빠르고 용량이 적음
# PCA로 훈련데이터의 차원을 축소하면 저장공간과 시간을 절약할 수 있음

# 앞에서 PCA 클래스 사용 시 n_components 매개변수에 주성분 개수를 50개로 진행하였음
# 이 대신 비율로 줄 수 있음 / ex) 50%
pca = PCA(n_components = 0.5)    # 주성분 50%(0 ~ 1 사이 실수로 입력하면 비율로 진행)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)

scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))    # 99.33%로 과대적합
print(np.mean(scores['fit_time']))    # 0.027초로 처리속도가 약간 느려짐

km = KMeans(n_clusters = 3, random_state = 42)
km.fit(fruits_pca)

for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

# 훈련데이터의 차원을 줄이면 또 하나 얻을 수 있는 장점은 시각화가 가능함
# 3개 이하로 차원을 줄이면 화면에 출력하기 비교적 쉬움
# fruits_pca 데이터가 2개의 특성으로 있기 떄문에 2차원으로 표현할 수 있음
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]    # km.labels_를 사용하여 클러스터별로 나누어 산점도를 그림
    plt.scatter(data[:, 0], data[:, 1])
plt.legend(['pineapple', 'banana', 'apple'])    # 범례표시
plt.show()