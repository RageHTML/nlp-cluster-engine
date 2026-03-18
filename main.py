from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

modelo = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Adiciona mais setencas aqui (siga a sintaxa)
setences = [
    "Hoje está chovendo muito",
    "O sol está forte hoje",
    "Amanhã vai fazer frio",
    "Eu vou viajar no fim de semana",
    "Preciso estudar para a prova",
    "Estou com muita fome agora",
    "Vou preparar o almoço",
    "Quero pedir comida pelo aplicativo",
    "Estou com sono e quero dormir",
    "Acordei muito cedo hoje",
    "Vou jogar videogame mais tarde",
    "Quero assistir um filme hoje à noite",
    "Ele foi trabalhar cedo",
    "Ela está estudando programação",
    "Preciso terminar meu projeto",
    "Vamos sair para caminhar",
    "Estou me sentindo cansado",
    "Quero aprender mais sobre inteligência artificial",
    "Vou comprar pão na padaria",
    "Estou organizando meus arquivos"
]

embeddings = modelo.encode(setences)

# num_cluster define a quantidade de cluster, nao coloque random state como 0
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

# Altere esta frase para testar em qual cluster sua sentença será classificada
new_sentence = "Programar e melhor do que copiar"
    
new_embedding = modelo.encode([new_sentence])
distances = np.linalg.norm(kmeans.cluster_centers_ - new_embedding, axis=1)
closest_cluster = np.argmin(distances)

print(f"\nA frase '{new_sentence}' pertence ao Cluster {closest_cluster}")