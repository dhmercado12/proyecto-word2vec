
import os
import sys
import time
import json
import hashlib
import platform
from gensim.models import Word2Vec

if platform.system() == "Windows":
    import psutil
else:
    import resource


def hash_estable(texto):
    return int.from_bytes(
        hashlib.sha256(texto.encode("utf-8")).digest()[:4],
        "little"
    )


class Corpus:
    def __init__(self, ruta):
        self.ruta = ruta

    def __iter__(self):
        with open(self.ruta, "r", encoding="utf-8", errors="replace") as archivo:
            for linea in archivo:
                palabras = linea.lower().split()
                if palabras:
                    yield palabras


def obtener_configuracion():
    linea = sys.stdin.readline().strip()

    if not linea:
        return {
            "corpus": input("Introduce la ruta del archivo TXT: ").strip().strip('"').strip("'")
        }

    try:
        datos = json.loads(linea)
        if isinstance(datos, dict):
            return datos
    except json.JSONDecodeError:
        pass

    return {"corpus": linea.strip('"').strip("'")}


def ejecutar():
    configuracion = obtener_configuracion()

    ruta = configuracion.get("corpus", "").strip().strip('"').strip("'")
    dimension = int(configuracion.get("dimension", 100))
    ventana = int(configuracion.get("ventana", 5))
    epocas = int(configuracion.get("epocas", 20))
    min_count = int(configuracion.get("min_count", 1))
    trabajadores = int(configuracion.get("trabajadores", 1))
    algoritmo = configuracion.get("algoritmo", "skipgram")

    if not os.path.isfile(ruta):
        print("Error: el archivo no existe.")
        raise SystemExit(1)

    if not 10 <= dimension <= 500:
        raise ValueError("La dimensión debe estar entre 10 y 500.")

    if not 1 <= ventana <= 20:
        raise ValueError("La ventana debe estar entre 1 y 20.")

    if not 1 <= epocas <= 100:
        raise ValueError("Las épocas deben estar entre 1 y 100.")

    if not 1 <= min_count <= 100:
        raise ValueError("La frecuencia mínima debe estar entre 1 y 100.")

    if not 1 <= trabajadores <= 32:
        raise ValueError("Los trabajadores deben estar entre 1 y 32.")

    if algoritmo not in ("skipgram", "cbow"):
        raise ValueError("El algoritmo debe ser skipgram o cbow.")

    ruta = os.path.abspath(ruta)
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    directorio = os.path.dirname(os.path.abspath(__file__))
    resultados = os.path.join(directorio, "resultados")
    os.makedirs(resultados, exist_ok=True)

    tamano = os.path.getsize(ruta)
    inicio_total = time.perf_counter()
    corpus = Corpus(ruta)

    modelo = Word2Vec(
        vector_size=dimension,
        window=ventana,
        min_count=min_count,
        workers=trabajadores,
        epochs=epocas,
        seed=42,
        sg=1 if algoritmo == "skipgram" else 0,
        negative=5,
        sample=0,
        shrink_windows=False,
        hashfxn=hash_estable
    )

    inicio = time.perf_counter()
    modelo.build_vocab(corpus_iterable=corpus)
    tiempo_vocabulario = time.perf_counter() - inicio

    oraciones = modelo.corpus_count
    palabras_totales = modelo.corpus_total_words
    vocabulario = len(modelo.wv)

    inicio = time.perf_counter()

    modelo.train(
        corpus_iterable=corpus,
        total_examples=modelo.corpus_count,
        epochs=epocas
    )

    tiempo_entrenamiento = time.perf_counter() - inicio

    ruta_modelo = os.path.join(
        resultados,
        f"modelo_secuencial_{nombre}.model"
    )

    inicio = time.perf_counter()
    modelo.save(ruta_modelo)
    tiempo_guardado = time.perf_counter() - inicio

    tiempo_total = time.perf_counter() - inicio_total

    if platform.system() == "Windows":
        memoria_mb = (
            psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        )
    else:
        memoria_mb = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )

    metricas = {
        "modo": "secuencial",
        "archivo": ruta,
        "tamano_bytes": tamano,
        "tamano_mb": round(tamano / (1024 ** 2), 3),
        "oraciones": int(oraciones),
        "palabras_totales": int(palabras_totales),
        "vocabulario": vocabulario,
        "dimension": dimension,
        "ventana": ventana,
        "epocas": epocas,
        "min_count": min_count,
        "muestras_negativas": 5,
        "algoritmo": algoritmo,
        "procesos": 1,
        "trabajadores": trabajadores,
        "tiempo_vocabulario_segundos": round(tiempo_vocabulario, 4),
        "tiempo_entrenamiento_segundos": round(tiempo_entrenamiento, 4),
        "tiempo_guardado_segundos": round(tiempo_guardado, 4),
        "tiempo_total_segundos": round(tiempo_total, 4),
        "palabras_por_segundo": round(
            palabras_totales / tiempo_entrenamiento, 2
        ) if tiempo_entrenamiento > 0 else 0,
        "memoria_maxima_mb": round(memoria_mb, 2),
        "modelo": ruta_modelo
    }

    ruta_metricas = os.path.join(
        resultados,
        f"metricas_secuencial_{nombre}.json"
    )

    with open(ruta_metricas, "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, ensure_ascii=False, indent=2)

    print("\nENTRENAMIENTO SECUENCIAL FINALIZADO")
    print("Archivo:", ruta)
    print("Algoritmo:", algoritmo)
    print("Dimensión:", dimension)
    print("Ventana:", ventana)
    print("Épocas:", epocas)
    print("Frecuencia mínima:", min_count)
    print("Trabajadores:", trabajadores)
    print("Tamaño (MB):", metricas["tamano_mb"])
    print("Oraciones:", oraciones)
    print("Palabras totales:", palabras_totales)
    print("Vocabulario:", vocabulario)
    print("Tiempo de vocabulario:", metricas["tiempo_vocabulario_segundos"], "s")
    print("Tiempo de entrenamiento:", metricas["tiempo_entrenamiento_segundos"], "s")
    print("Tiempo de guardado:", metricas["tiempo_guardado_segundos"], "s")
    print("Tiempo total:", metricas["tiempo_total_segundos"], "s")
    print("Palabras por segundo:", metricas["palabras_por_segundo"])
    print("Memoria:", metricas["memoria_maxima_mb"], "MB")
    print("Modelo guardado:", ruta_modelo)
    print("Métricas guardadas:", ruta_metricas)


if __name__ == "__main__":
    ejecutar()
