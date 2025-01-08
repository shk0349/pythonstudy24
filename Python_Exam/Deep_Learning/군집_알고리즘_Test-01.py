import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# url에서 파일 다운로드
url = "https://bit.ly/fruits_300_data"
output_file = "fruits_300.npy"

try:
    urllib.request.urlretrieve(url, output_file)
    print(f"File download is successful: {output_file}")
except Exception as e:
    print(f"File download is failed: {e}")

fruits = np.load('fruits_300.npy')

# 인덱스 0에 저정된 이미지 출력(흑백사진 반전)
plt.imshow(fruits[0], cmap = 'gray')
plt.show()

# 반전된 이미지 반전출력
plt.imshow(fruits[0], cmap = 'gray_r')
plt.show()

# 50, 100, 200번 인덱스 이미지 출력
fig, axs = plt.subplots(1, 3)    # 1행 3열 / 반환되는 axs는 3개의 서브그래프를 담고 있음
axs[0].imshow(fruits[0], cmap = 'gray_r')
axs[1].imshow(fruits[100], cmap = 'gray_r')
axs[2].imshow(fruits[200], cmap = 'gray_r')
plt.show()

# 2차원 배열을 1차원 배열로 변환
apple =fruits[0:100].reshape(-1, 100 * 100)
pineapple =fruits[100:200].reshape(-1, 100 * 100)
banana =fruits[200:300].reshape(-1, 100 * 100)

# 히스토그램 : 값이 발생한 빈도를 그래프로 표시
# axis = 0 : 행을 따라 계산 / axis = 1 : 열을 따라 계산
plt.hist(np.mean(apple, axis = 1), alpha = 0.8)    # alpha : 투명도
plt.hist(np.mean(pineapple, axis = 1), alpha = 0.8)
plt.hist(np.mean(banana, axis = 1), alpha = 0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.xlabel('panel')
plt.ylabel('frequency')
plt.show()
# 결론
# 바나나 : 40점 부근에 주로 분포 / 사과, 파인애플 : 90점 부근에 주로 분포
# 둥근 사과나 파인애플에 비해 길이가 긴 바나나는 평균 분포도가 다름
# 이는 바나나가 길이가 길어 빈여백이 많고, 둥근 사과나 파인애플은 여백이 상대적으로 적기 때문

# 위 그래프 개선 필요 / axis = 1로 변경 및 투명도 하향 필요
plt.hist(np.mean(apple, axis = 0), alpha = 0.5)    # alpha : 투명도
plt.hist(np.mean(pineapple, axis = 0), alpha = 0.5)
plt.hist(np.mean(banana, axis = 0), alpha = 0.5)
plt.legend(['apple', 'pineapple', 'banana'])
plt.xlabel('panel')
plt.ylabel('frequency')
plt.show()

# 픽셀별 평균값 비교
fig, axs = plt.subplots(1, 3, figsize = (20, 5))
axs[0].bar(range(10000), np.mean(apple, axis = 0))
axs[1].bar(range(10000), np.mean(apple, axis = 0))
axs[2].bar(range(10000), np.mean(apple, axis = 0))
plt.show()

# 픽셀의 평균값을 100 x 100으로 변경하여 이미지 출력
apple_mean = np.mean(apple, axis = 0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis = 0).reshape(100, 100)
banana_mean = np.mean(banana, axis = 0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize = (20 ,5))
axs[0].imshow(apple_mean, cmap = 'gray_r')
axs[1].imshow(pineapple_mean, cmap = 'gray_r')
axs[2].imshow(banana_mean, cmap = 'gray_r')
plt.show()

# 평균값과 가까운 사진 고르기
# 사과에 가까운 이미지 출력
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
# 배열에 사용하면 모든 원소의 절대값을 계산하여 입력과 동일한 크기의 배열 반환
# numpy의 absolute()와 유사한 기능
apple_index = np.argsort(abs_mean)[:100]
# argsort(abs_mean)[:100] : 작은것에서 큰 순서대로 나열한 abs_mean 배열의 인덱스 반환
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
# subplots() : 서브그래프 생성
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap = 'gray_r')
        axs[i, j].axis('off')    # axis('off') : 깔끔한 이미지만 표시
plt.show()

# 파인애플에 가까운 이미지 출력
abs_diff = np.abs(fruits - pineapple_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
pineapple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[pineapple_index[i * 10 + j]], cmap = 'gray_r')
        axs[i, j].axis('off')
plt.show()

# 바나나에 가까운 이미지 출력
abs_diff = np.abs(fruits - banana_mean)
abs_mean = np.mean(abs_diff, axis = (1, 2))
banana_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[banana_index[i * 10 + j]], cmap = 'gray_r')
        axs[i, j].axis('off')
plt.show()

# 클러스터링(군집) : 비슷한 샘플끼리 그룹으로 모으는 작업
# 위 작업은 사과, 파인애플, 바나나가 있다는 전제가 있었으므로 타깃 값을 알고 접근
# 그러나 실제 비지도학습에서는 타깃값을 모르는 것이 대부분 -> 샘플 평균값을 미리 구할 수 없음