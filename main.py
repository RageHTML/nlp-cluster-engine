from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

modelo = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


setences = [
    "Hoje esta chovendo",
    "Amanha vai ser sol",
    "Vou jogar mais tarde",
    "Eu nao quero almocar",
    "Estou com fome",
    "Estou com muito sono",
    "Agora eu vou dormir",
    "Eu queria jogar mas tenho que estudar"
]
embeddings = modelo.encode(setences)

num_cluster = 2
kmeans = KMeans(n_clusters=num_cluster,random_state=1)
kmeans.fit(embeddings)
labels = kmeans.labels_

clusters = {}
for sentence, label in zip(setences,labels):
    clusters.setdefault(label,[]).append(sentence)


for cluster_id, cluster_sentences in clusters.items():
    print(f"Cluster {cluster_id}:")
    for s in cluster_sentences:
        print("  -", s)

new_sentence = "O sol está forte hoje"
new_embedding = modelo.encode([new_sentence])
distances = np.linalg.norm(kmeans.cluster_centers_ - new_embedding, axis=1)
closest_cluster = np.argmin(distances)

print(f"\nA frase '{new_sentence}' pertence ao Cluster {closest_cluster}")