import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# dataset 폴더 내 모든 csv 파일 읽기
dataset_dir = 'dataset'
all_questions = []

for fname in os.listdir(dataset_dir):
    if fname.endswith('.csv'):
        df = pd.read_csv(os.path.join(dataset_dir, fname))
        # 'question' 컬럼이 있다고 가정
        if 'question' in df.columns:
            all_questions.extend(df['question'].dropna().tolist())

# 벡터화
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(all_questions)

# 클러스터링 (예: 5개 클러스터)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# 결과 출력
for idx, (question, label) in enumerate(zip(all_questions, labels)):
    print(f"[Cluster {label}] {question}")

# --- 클러스터 결과 시각화 ---
# 2차원으로 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.title('KMeans Clustering of Questions (PCA Visualization)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')
plt.show()