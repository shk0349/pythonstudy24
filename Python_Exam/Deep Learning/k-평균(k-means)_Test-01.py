import urllib.request
import numpy as np
import matplotlib.pyplot as plt
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

# k-means : 군집 알고리즘이 비지도학습의 평균값을 자동으로 알려줌
km = KMeans(n_clusters = 3)
km.fit(fruits_2d)
print("k-means 적용 확률")
print(np.unique(km.labels_, return_counts = True))
print("=" * 50)

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

draw_fruits(fruits[km.labels_ == 0])    # 바나나와 비슷한 이미지 출력
draw_fruits(fruits[km.labels_ == 1])    # 사과와 비슷한 이미지 출력
draw_fruits(fruits[km.labels_ == 2])    # 파인애플과 비슷한 이미지 출력
# 파인애플에서 오류가 약간 발생함 / 훈련데이터에 타깃 레이블을 미제공하였음에도 확률은 나쁘지 않음

# fruits_2d 샘플의 클러스터 중심을 이미지 출력
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio = 4)

# transform() 매서드 활용 훈련데이터 샘플에서 클러스터 중심까지 거리 변환
print("transform 적용 100번째 이미지 클러스터 중심까지 거리")
print(km.transform(fruits_2d[100:101]))
print("=" * 50)

# predict() 매서드 활용 가장 가까운 클러스터 중심을 예측 클래스로 이용
# 0 : 파인애플 / 1 : 바나나 / 2 : 사과
print("predict 활용 0번쨰 이미지 예측")
print(km.predict(fruits_2d[0:1]))
print("=" * 50)

print("predict 활용 100번쨰 이미지 예측")
print(km.predict(fruits_2d[100:101]))
print("=" * 50)

print("predict 활용 200번쨰 이미지 예측")
print(km.predict(fruits_2d[200:201]))
print("=" * 50)

# 0, 100, 200번째 이미지 출력
draw_fruits(fruits[0:1])    # 0번째 이미지 출력
draw_fruits(fruits[100:101])    # 100번째 이미지 출력
draw_fruits(fruits[200:201])    # 200번째 이미지 출력

# n_iter_ : 반복횟수 / 활용 반복적으로 클러스터 중심을 옮겨가며 최적의 클러스터 찾기

# 지금까지는 3개의 객체라는 것을 편법으로 적용하여 분석을 하였지만 실무에서는 클러스터 개수 조차 모름(n_cluster 3)
# 최적의 k 찾기 -> 많은 기법이 있지만 엘보우를 사용하여 분석

# k-평균 알고리즘은 클러스터 중심과 클러스터에 속한 샘플 사이의 거리를 잴 수 있는데, 이 거리의 제곱 합을 이너셔(inertia)라고 함
# inertia : 클러스터에 속한 샘플이 얼마나 가깝게 모여있는지 나타내는 값(클러스터의 샘플이 얼마나 가깝게 있는지를 나타내는 값)
# 일반적으로 클러스터 갯수가 늘어나면 클러스터 개개의 크기는 줄어들기 떄문에 이너셔도 줄어듬
# 엘보우의 방법은 클러스터 개수를 늘려가면서 이너셔의 변화를 관찰하여 최적의 클러스터 개수를 찾음

# 클러스터 개수를 증가시키면서 이너셔를 그래프로 그리면 감소하는 속도가 꺾이는 지점이 있는데
# 이때부터 클러스터 개수를 늘려도 클러스터에 잘 밀집된 정도가 크게 개선되지 않음 -> 이너셔가 크게 줄어들지 않음
# 그래서 팔꿈치 모양이라고 해서 엘보우 방법이라 함

# KMeans 클래스에서 자동으로 이너셔를 계산하는 inertia_ 속성이 있음
# 클러스터 계수 k를 2 ~ 6까지 변경하면서 5번 훈련

inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters = k, n_init = 'auto')
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
# k=3 지점에서 꺽이는 엘보우 현상 발현 -> 3개의 군집으로 만들면 됨