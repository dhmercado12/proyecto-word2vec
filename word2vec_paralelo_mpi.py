import argparse
import os
import time
import json
import hashlib
import platform
from gensim.models import Word2Vec
from mpi4py import MPI

if platform.system() == "Windows":
    import psutil
else:
    import resource
comunicador = MPI.COMM_WORLD
rango = comunicador.Get_rank()
total = comunicador.Get_size()

dimension = 100
ventana = 5
epocas = 20
negativos = 5
min_count = 1
algoritmo = "skipgram"

def hash_estable(texto):
    return int.from_bytes(
        hashlib.sha256(texto.encode("utf-8")).digest()[:4],
        "little"
    )

def leer_fragmento(ruta, rango, total):
    tamano = os.path.getsize(ruta)
    inicio = tamano * rango // total
    fin = tamano * (rango + 1) // total

    with open(ruta, "rb") as archivo:
        if inicio > 0:
            archivo.seek(inicio - 1)
            anterior = archivo.read(1)

            if anterior == b"\n":
                archivo.seek(inicio)
            else:
                archivo.readline()

        while archivo.tell() < fin:
            linea = archivo.readline()

            if not linea:
                break

            palabras = linea.decode(
                "utf-8", errors="replace"
            ).lower().split()

            if palabras:
                yield palabras

if rango == 0:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    argumentos = parser.parse_args()

    with open(argumentos.config, "r", encoding="utf-8") as archivo:
        configuracion = json.load(archivo)
else:
    configuracion = None

configuracion = comunicador.bcast(configuracion, root=0)

ruta_config = configuracion["corpus"]

if isinstance(ruta_config, dict):
    ruta = (
        ruta_config.get("ruta")
        or ruta_config.get("path")
        or ruta_config.get("filepath")
    )
else:
    ruta = ruta_config

if not isinstance(ruta, str) or not ruta:
    raise ValueError(
        f"Ruta del corpus no válida: {ruta_config!r}"
    )
nombre_resultado = configuracion["nombre"]

if isinstance(ruta, dict):
    ruta = ruta.get("ruta") or ruta.get("path")
dimension = int(configuracion["dimension"])
ventana = int(configuracion["ventana"])
epocas = int(configuracion["epocas"])
min_count = int(configuracion["min_count"])
algoritmo = configuracion["algoritmo"]

if algoritmo not in ("skipgram", "cbow"):
    raise ValueError("Algoritmo no válido")

sg = 1 if algoritmo == "skipgram" else 0

archivos_validos = comunicador.allgather(os.path.isfile(ruta))

if not all(archivos_validos):
    if rango == 0:
        print("Error: el archivo debe existir en todos los nodos.")
    raise SystemExit(1)

ruta = os.path.abspath(ruta)
tamano = os.path.getsize(ruta)
tamanos = comunicador.allgather(tamano)

if len(set(tamanos)) != 1:
    if rango == 0:
        print("Error: el archivo tiene tamaños diferentes entre nodos.")
    raise SystemExit(1)

nombre = nombre_resultado
directorio = os.path.dirname(os.path.abspath(__file__))
resultados = os.path.join(directorio, "resultados")

if rango == 0:
    os.makedirs(resultados, exist_ok=True)

comunicador.Barrier()
inicio_total = MPI.Wtime()
inicio_vocabulario = MPI.Wtime()

frecuencias_locales = Counter()
oraciones_locales = 0
palabras_locales = 0

for palabras in leer_fragmento(ruta, rango, total):
    frecuencias_locales.update(palabras)
    oraciones_locales += 1
    palabras_locales += len(palabras)

frecuencias_por_proceso = comunicador.gather(
    dict(frecuencias_locales), root=0
)

if rango == 0:
    frecuencias_globales = Counter()

    for frecuencias in frecuencias_por_proceso:
        frecuencias_globales.update(frecuencias)

    frecuencias_globales = dict(frecuencias_globales)
else:
    frecuencias_globales = None

frecuencias_globales = comunicador.bcast(
    frecuencias_globales, root=0
)

oraciones_totales = comunicador.allreduce(
    oraciones_locales, op=MPI.SUM
)

palabras_totales = comunicador.allreduce(
    palabras_locales, op=MPI.SUM
)

modelo = Word2Vec(
    vector_size=dimension,
    window=ventana,
    min_count=min_count,
    workers=1,
    epochs=epocas,
    seed=42,
    sg=sg,
    negative=negativos,
    sample=0,
    shrink_windows=False,
    hashfxn=hash_estable
)

modelo.build_vocab_from_freq(frecuencias_globales)

comunicador.Barrier()

tiempo_vocabulario = comunicador.allreduce(
    MPI.Wtime() - inicio_vocabulario, op=MPI.MAX
)

oraciones_por_proceso = comunicador.allgather(oraciones_locales)
palabras_por_proceso = comunicador.allgather(palabras_locales)
nodos = comunicador.allgather(socket.gethostname())

comunicador.Barrier()
inicio_entrenamiento = MPI.Wtime()

for epoca in range(epocas):
    alpha_inicio = 0.025 * (1 - epoca / epocas)
    alpha_fin = max(
        0.0001, 0.025 * (1 - (epoca + 1) / epocas)
    )

    if oraciones_locales > 0:
        modelo.train(
            corpus_iterable=leer_fragmento(ruta, rango, total),
            total_examples=oraciones_locales,
            epochs=1,
            start_alpha=alpha_inicio,
            end_alpha=alpha_fin
        )

    comunicador.Allreduce(
        MPI.IN_PLACE, modelo.wv.vectors, op=MPI.SUM
    )
    modelo.wv.vectors /= total

    comunicador.Allreduce(
        MPI.IN_PLACE, modelo.syn1neg, op=MPI.SUM
    )
    modelo.syn1neg /= total

comunicador.Barrier()

tiempo_entrenamiento = comunicador.allreduce(
    MPI.Wtime() - inicio_entrenamiento, op=MPI.MAX
)

memoria_local = (
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
)

memorias = comunicador.allgather(memoria_local)

ruta_modelo = os.path.join(
    resultados, f"modelo_paralelo_mpi_{nombre}.model"
)

tiempo_guardado = 0

if rango == 0:
    inicio_guardado = MPI.Wtime()
    modelo.save(ruta_modelo)
    tiempo_guardado = MPI.Wtime() - inicio_guardado

tiempo_guardado = comunicador.bcast(tiempo_guardado, root=0)

comunicador.Barrier()

tiempo_total = comunicador.allreduce(
    MPI.Wtime() - inicio_total, op=MPI.MAX
)

if rango == 0:
    metricas = {
        "archivo": ruta,
        "tamano_bytes": tamano,
        "tamano_mb": round(tamano / (1024 ** 2), 3),
        "oraciones": int(oraciones_totales),
        "palabras_totales": int(palabras_totales),
        "vocabulario": len(modelo.wv),
        "dimension": dimension,
        "ventana": ventana,
        "epocas": epocas,
        "muestras_negativas": negativos,
        "procesos": total,
        "nodos": nodos,
        "oraciones_por_proceso": oraciones_por_proceso,
        "palabras_por_proceso": palabras_por_proceso,
        "tiempo_vocabulario_segundos": round(tiempo_vocabulario, 4),
        "tiempo_entrenamiento_segundos": round(tiempo_entrenamiento, 4),
        "tiempo_guardado_segundos": round(tiempo_guardado, 4),
        "tiempo_total_segundos": round(tiempo_total, 4),
        "palabras_por_segundo": round(
            palabras_totales / tiempo_entrenamiento, 2
        ) if tiempo_entrenamiento > 0 else 0,
        "memoria_maxima_por_proceso_mb": [
            round(m, 2) for m in memorias
        ],
        "memoria_maxima_mb": round(max(memorias), 2),
        "memoria_acumulada_picos_mb": round(sum(memorias), 2),
        "modelo": ruta_modelo
    }

    ruta_metricas = os.path.join(
        resultados, f"metricas_paralelo_mpi_{nombre}.json"
    )

    with open(ruta_metricas, "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, ensure_ascii=False, indent=2)

    print("\nENTRENAMIENTO PARALELO MPI FINALIZADO")
    print("Archivo:", ruta)
    print("Tamaño (MB):", metricas["tamano_mb"])
    print("Oraciones:", oraciones_totales)
    print("Palabras totales:", palabras_totales)
    print("Vocabulario:", len(modelo.wv))
    print("Procesos MPI:", total)

    for i in range(total):
        print(
            f"Nodo {nodos[i]}: "
            f"{oraciones_por_proceso[i]} oraciones, "
            f"{palabras_por_proceso[i]} palabras"
        )

    print("Tiempo de vocabulario:", metricas["tiempo_vocabulario_segundos"], "s")
    print("Tiempo de entrenamiento:", metricas["tiempo_entrenamiento_segundos"], "s")
    print("Tiempo de guardado:", metricas["tiempo_guardado_segundos"], "s")
    print("Tiempo total:", metricas["tiempo_total_segundos"], "s")
    print("Palabras por segundo:", metricas["palabras_por_segundo"])
    print("Memoria máxima por proceso (MB):", memorias)
    print("Modelo guardado:", ruta_modelo)
    print("Métricas guardadas:", ruta_metricas)

