import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


dataset_dir = '../dataset/QA'
# domain_labels = [
#     "Military Strategy", "Weapon Systems", "Military Organization", "Military Law", "Military History",  # Defense Domain, 
#     "Medicine", "Law", "Economics", "Science", "IT",                                              # Non-Defense Domain
#     "Daily Knowledge", "Basic Knowledge", "Difficulty"      # Noise Data
# ]
domain_labels = [
    "Defense",  # Defense Domain, 
    "Non-Defense" # Non-Defense Domain
    #"Noise"      # Noise Data
]
num_clusters = len(domain_labels)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def visualize_clusters(sentence_embeddings, labels, num_clusters, cluster_to_domain):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(sentence_embeddings.detach().cpu().numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', alpha=0.7)
    plt.title('KMeans Clustering of Questions (PCA Visualization)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [cluster_to_domain[i] for i in range(num_clusters)], 
            title="Domains", 
            loc='center left', 
            bbox_to_anchor=(1.05, 0.5),  # 플롯 바깥 오른쪽에 배치
            borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('kmeans_clusters.png')
    plt.show()

def representative_questions(sentence_embeddings, labels, num_clusters, all_questions, cluster_to_domain, kmeans):
    cluster_centers = kmeans.cluster_centers_

    # 각 질문과 클러스터 중심 간의 거리 계산
    distances = []
    for i in range(len(sentence_embeddings)):
        cluster_label = labels[i]
        center = cluster_centers[cluster_label]
        dist = np.linalg.norm(sentence_embeddings[i].detach().cpu().numpy() - center)
        distances.append((i, dist))

    # 각 클러스터에서 대표 질문 선택
    representative_questions = {}
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        if cluster_indices: 
            cluster_distances = [(idx, dist) for idx, dist in distances if idx in cluster_indices]
            representative_idx = sorted(cluster_distances, key=lambda x: x[1])[:20]
            representative_questions[cluster_id] = [all_questions[idx] for idx, _ in representative_idx]
   
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    for cluster_id, question in representative_questions.items():
        count = cluster_counts.get(cluster_id, 0)
        print(f"[Cluster {cluster_id} - {cluster_to_domain[cluster_id]}] Representative Question: {question} | Number of questions: {count}")

def main():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    all_questions = []
    for fname in os.listdir(dataset_dir):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(dataset_dir, fname))
            if 'input' in df.columns:
                all_questions.extend(df['input'].dropna().tolist())
    
    # 군사 데이터 제외
    # all_questions = all_questions[2000:]

    # Tokenize sentences
    encoded_input = tokenizer(all_questions, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(sentence_embeddings)

    cluster_to_domain = {i: domain_labels[i] for i in range(num_clusters)}

    visualize_clusters(sentence_embeddings, labels, num_clusters, cluster_to_domain)
    representative_questions(sentence_embeddings, labels, num_clusters, all_questions, cluster_to_domain, kmeans)

if __name__ == "__main__":
    main()
