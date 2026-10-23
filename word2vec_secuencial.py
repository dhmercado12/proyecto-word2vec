import time
from gensim.models import Word2Vec

inicio = time.time()

with open("datos/corpus_prueba.txt", "r", encoding="utf-8") as archivo:
    oraciones = [linea.lower().split() for linea in archivo if linea.strip()]

modelo = Word2Vec(
    sentences=oraciones,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    epochs=20,
    seed=42
)

modelo.save("resultados/modelo_secuencial.model")

fin = time.time()

print("Entrenamiento secuencial finalizado")
print("Cantidad de oraciones:", len(oraciones))
print("Palabras en el vocabulario:", len(modelo.wv))
print("Tiempo de ejecucion:", round(fin - inicio, 4), "segundos")
print("Vector de ejemplo para 'procesamiento':")
print(modelo.wv["procesamiento"])
