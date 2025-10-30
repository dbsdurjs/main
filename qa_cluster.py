import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch, json
import numpy as np

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
    # Defense 클러스터 ID 찾기
    defense_cluster_ids = [cid for cid in cluster_to_domain if cluster_to_domain[cid] == "Defense"]
    
    if not defense_cluster_ids:
        print("Defense 클러스터를 찾지 못했습니다.")
        return
    
    # Defense 클러스터에 속한 모든 질문 인덱스 수집
    defense_indices = [i for i, label in enumerate(labels) if label in defense_cluster_ids]
    
    # {prompt: question} 형식으로 저장
    defense_data = [all_questions[i] for i in defense_indices]  # 키를 'prompt'로 맞춤 (질문이 input이므로)
    
    file_name = 'defense_questions_snunlp.txt'
    # TXT 파일로 저장 (각 dict를 JSON 문자열로 한 줄씩)
    with open(file_name, 'w', encoding='utf-8') as f:
        for item in defense_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Defense 클러스터 질문 {len(defense_data)}개가 {file_name}에 저장되었습니다.")

#-------------------------------------------------------------------------------------------------------------------------------
# 평가 데이터가 적기 때문에 우선은 클러스터링 된 데이터들을 모두 사용
# 대표 질문들을 사용하기 위한 코드

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
#-------------------------------------------------------------------------------------------------------------------------------

def main():
    tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    model = AutoModel.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    all_questions = []
    for fname in os.listdir(dataset_dir):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(dataset_dir, fname))
            if 'input' in df.columns:
                all_questions.extend(df['input'].dropna().tolist())

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
