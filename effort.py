import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
lev0=pd.read_csv("노인 치매 검사 데이터_정상.csv",encoding='cp949',skiprows=1,header=None).to_numpy()
lev1=pd.read_csv("노인 치매 검사 데이터_치매 의심.csv",encoding='cp949',skiprows=1,header=None).to_numpy()
lev2=pd.read_csv("노인 치매 검사 데이터_경증 치매.csv",encoding='cp949',skiprows=1,header=None).to_numpy()
lev3=pd.read_csv("노인 치매 검사 데이터_중증 치매.csv",encoding='cp949',skiprows=1,header=None).to_numpy()
#JN:지남력, KU:기억력, JW:주의력, UU:언어능력, PD:판단력
LEV0_para={'JN':[],'KU':[],'UU':[],'PD':[]}
LEV1_para={'JN':[],'KU':[],'UU':[],'PD':[]}
LEV2_para={'JN':[],'KU':[],'UU':[],'PD':[]}
LEV3_para={'JN':[],'KU':[],'UU':[],'PD':[]}
#받은 파일, 파일 번호, 시작질문 위치, 끝질문 위치
def PARA(file0,N):
    var_name="para_"+str(N)
    globals()[var_name]={'JN':[],'KU':[],'UU':[],'PD':[]}
    #지남력은 1:2, 기억력은 3:4, 언어능력은 5:6, 판단력은 7:8열에서 점수가 있다고 가정한다.
    parameterstarting={'JN':(1,2),'KU':(3,4),'UU':(5,6),'PD':(7,8)}
    for i in file0:
        for j in globals()[var_name]:
            globals()[var_name][j].append(list(i[parameterstarting[j][0]:parameterstarting[j][1]]).count('O')/2)
PARA(lev0,0)
PARA(lev1,1)
PARA(lev2,2)
PARA(lev3,3)
lev0_df = pd.DataFrame(para_0).assign(cluster=0)
lev1_df = pd.DataFrame(para_1).assign(cluster=1)
lev2_df = pd.DataFrame(para_2).assign(cluster=2)
lev3_df = pd.DataFrame(para_3).assign(cluster=3)
data = pd.concat([lev0_df, lev1_df, lev2_df, lev3_df])
# 피처와 타겟 나누기
X = data.drop('cluster', axis=1)
y = data['cluster']
# 학습 데이터와 테스트 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# KNN 모델 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# 새로운 데이터에 대한 예측
new_data = pd.DataFrame([[0.7, 0.8, 0.7, 0.6]], columns=X.columns)  # 새로운 데이터.일단 가정한다.
prediction = knn.predict(new_data)
print("예측된 군집:", prediction)
# PCA로 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
new_data_pca = pca.transform(new_data)
# 각 군집에 대해 scatter plot 생성
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.3)  # alpha 값을 낮게 설정
# 각 군집에 레이블 표시
for cluster in np.unique(y):
    plt.scatter([], [], alpha=0.3, label=cluster)
plt.legend(title='Cluster')
# 새로운 데이터 포인트 표시
plt.scatter(new_data_pca[:, 0], new_data_pca[:, 1], c='red', marker='x', label='New Data')
plt.legend()
# 그래프의 x와 y 축 범위를 데이터에 맞게 조정
plt.xlim(min(X_pca[:, 0].min(), new_data_pca[:, 0].min()), max(X_pca[:, 0].max(), new_data_pca[:, 0].max()))
plt.ylim(min(X_pca[:, 1].min(), new_data_pca[:, 1].min()), max(X_pca[:, 1].max(), new_data_pca[:, 1].max()))
plt.show()
