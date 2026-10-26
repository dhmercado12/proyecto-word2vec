from flask import Flask, render_template, jsonify, request
from pathlib import Path
import json
import subprocess
import sys
import threading
import time

app = Flask(__name__)

CARPETA = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA / "datos"
CARPETA_RESULTADOS = CARPETA / "resultados"
SCRIPT_SECUENCIAL = CARPETA / "word2vec_secuencial_stream.py"
HOST_MPI = "mpiuser@192.168.44.133"
CARPETA_MPI = "/head/NFS/mpiuser/proyecto_word2vec"
PYTHON_MPI = f"{CARPETA_MPI}/word2vec-env/bin/python"
MPIRUN = "/opt/openmpi/4.1.5/bin/mpirun"
HOSTFILE_MPI = f"{CARPETA_MPI}/nodos.txt"
SCRIPT_MPI = f"{CARPETA_MPI}/word2vec_paralelo_mpi.py"
PROCESOS_MPI = 2

trabajo = {
    "activo": False,
    "estado": "Listo",
    "salida": "",
    "inicio": None,
    "fin": None,
    "error": ""
}
candado = threading.Lock()


def obtener_corpus():
    """Devuelve los archivos TXT disponibles dentro de datos/."""
    if not CARPETA_DATOS.exists():
        return []

    archivos = []
    for archivo in CARPETA_DATOS.glob("*.txt"):
        archivos.append({
            "nombre": archivo.name,
            "ruta": f"datos/{archivo.name}",
            "bytes": archivo.stat().st_size
        })
    return sorted(archivos, key=lambda x: x["nombre"])


def leer_metricas():
    """Lee las métricas JSON generadas por los programas Word2Vec."""
    resultados = []
    if not CARPETA_RESULTADOS.exists():
        return resultados

    for archivo in sorted(
        CARPETA_RESULTADOS.glob("metricas_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    ):
        try:
            with archivo.open("r", encoding="utf-8") as f:
                datos = json.load(f)
            datos["_archivo"] = archivo.name
            datos["_modificado"] = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(archivo.stat().st_mtime)
            )
            resultados.append(datos)
        except (OSError, json.JSONDecodeError):
            continue

    return resultados



def ejecutar_secuencial(configuracion):
    try:
        with candado:
            trabajo["activo"] = True
            trabajo["estado"] = "Ejecutando Word2Vec secuencial"
            trabajo["salida"] = ""
            trabajo["inicio"] = time.time()
            trabajo["fin"] = None
            trabajo["error"] = ""

        proceso = subprocess.Popen(
            [sys.executable, str(SCRIPT_SECUENCIAL)],
            cwd=str(CARPETA),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        proceso.stdin.write(json.dumps(configuracion) + "\n")
        proceso.stdin.flush()
        proceso.stdin.close()

        for linea in proceso.stdout:
            with candado:
                trabajo["salida"] += linea

        codigo = proceso.wait()

        with candado:
            trabajo["activo"] = False
            trabajo["fin"] = time.time()

            if codigo == 0:
                trabajo["estado"] = "Entrenamiento finalizado"
            else:
                trabajo["estado"] = "El entrenamiento terminó con error"
                trabajo["error"] = f"El programa terminó con código {codigo}."

    except Exception as error:
        with candado:
            trabajo["activo"] = False
            trabajo["fin"] = time.time()
            trabajo["estado"] = "No se pudo iniciar el entrenamiento"
            trabajo["error"] = str(error)


def ejecutar_paralelo(configuracion):
    nombre = f"web_{int(time.time())}"
    corpus_local = CARPETA_RESULTADOS / f"corpus_{nombre}.txt"
    config_local = CARPETA_RESULTADOS / f"config_{nombre}.json"

    corpus_remoto = f"{CARPETA_MPI}/datos/corpus_{nombre}.txt"
    config_remoto = f"{CARPETA_MPI}/config_{nombre}.json"
    ruta_metricas_remota = f"{CARPETA_MPI}/resultados/metricas_paralelo_mpi_{nombre}.json"
    ruta_modelo_remota = f"{CARPETA_MPI}/resultados/modelo_paralelo_mpi_{nombre}.model"

    try:
        with candado:
            trabajo["activo"] = True
            trabajo["estado"] = "Preparando corpus para MPI"
            trabajo["salida"] = ""
            trabajo["inicio"] = time.time()
            trabajo["fin"] = None
            trabajo["error"] = ""

        with corpus_local.open("wb") as salida:
            for ruta in configuracion["corpus"]:
                with open(ruta, "rb") as entrada:
                    contenido = entrada.read()
                salida.write(contenido)
                if contenido and not contenido.endswith(b"\n"):
                    salida.write(b"\n")

        configuracion_mpi = {
            "corpus": corpus_remoto,
            "nombre": nombre,
            "dimension": configuracion["dimension"],
            "ventana": configuracion["ventana"],
            "epocas": configuracion["epocas"],
            "min_count": configuracion["min_count"],
            "algoritmo": configuracion["algoritmo"],
            "archivos_originales": [
                Path(ruta).name for ruta in configuracion["corpus"]
            ]
        }

        with config_local.open("w", encoding="utf-8") as archivo:
            json.dump(configuracion_mpi, archivo, ensure_ascii=False, indent=2)

        comandos_copia = [
            ["scp", str(corpus_local), f"{HOST_MPI}:{corpus_remoto}"],
            ["scp", str(config_local), f"{HOST_MPI}:{config_remoto}"]
        ]

        for comando in comandos_copia:
            resultado = subprocess.run(
                comando,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120
            )
            if resultado.returncode != 0:
                raise RuntimeError(resultado.stderr or "Falló la transferencia por SCP.")

        with candado:
            trabajo["estado"] = "Ejecutando entrenamiento paralelo MPI"

        comando_mpi = (
    f"{MPIRUN} --hostfile {HOSTFILE_MPI} "
    f"-np {PROCESOS_MPI} "
    f"{PYTHON_MPI} {SCRIPT_MPI} --config {config_remoto}"
)

        proceso = subprocess.Popen(
            ["ssh", HOST_MPI, comando_mpi],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        for linea in proceso.stdout:
            with candado:
                trabajo["salida"] += linea

        codigo = proceso.wait()

        if codigo != 0:
            raise RuntimeError(
                f"MPI terminó con código {codigo}. Revisa la salida del proceso."
            )

        for ruta_remota, ruta_local in [
            (
                ruta_metricas_remota,
                CARPETA_RESULTADOS / f"metricas_paralelo_mpi_{nombre}.json"
            ),
            (
                ruta_modelo_remota,
                CARPETA_RESULTADOS / f"modelo_paralelo_mpi_{nombre}.model"
            )
        ]:
            resultado = subprocess.run(
                ["scp", f"{HOST_MPI}:{ruta_remota}", str(ruta_local)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300
            )
            if resultado.returncode != 0:
                raise RuntimeError(
                    resultado.stderr or "No se pudieron recuperar los resultados de MPI."
                )

        with candado:
            trabajo["estado"] = "Entrenamiento paralelo finalizado"

    except Exception as error:
        with candado:
            trabajo["estado"] = "El entrenamiento paralelo terminó con error"
            trabajo["error"] = str(error)

    finally:
        try:
            subprocess.run(
                [
                    "ssh", HOST_MPI,
                    f"rm -f '{corpus_remoto}' '{config_remoto}'"
                ],
                capture_output=True,
                timeout=30
            )
        except Exception:
            pass

        for archivo in (corpus_local, config_local):
            try:
                archivo.unlink(missing_ok=True)
            except OSError:
                pass

        with candado:
            trabajo["activo"] = False
            trabajo["fin"] = time.time()


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/api/inicial")
def api_inicial():
    return jsonify({
        "corpus": obtener_corpus(),
        "metricas": leer_metricas(),
        "estado": estado_trabajo()
    })


def estado_trabajo():
    with candado:
        copia = dict(trabajo)
    if copia["inicio"] is not None:
        fin = copia["fin"] if copia["fin"] is not None else time.time()
        copia["duracion_segundos"] = round(fin - copia["inicio"], 1)
    else:
        copia["duracion_segundos"] = 0
    return copia



@app.route("/api/estado")
def api_estado():
    return jsonify({
        "corpus": obtener_corpus(),
        "estado": estado_trabajo(),
        "metricas": leer_metricas()
    })





@app.route("/api/ejecutar", methods=["POST"])
def api_ejecutar():
    datos = request.get_json(silent=True) or {}
    modo = datos.get("modo", "secuencial")
    rutas = datos.get("corpus", [])

    if isinstance(rutas, str):
        rutas = [rutas]

    if not rutas or not all(isinstance(ruta, str) for ruta in rutas):
        return jsonify({
            "ok": False,
            "mensaje": "Selecciona uno o varios archivos TXT."
        }), 400

    if modo not in ("secuencial", "paralelo"):
        return jsonify({
            "ok": False,
            "mensaje": "Selecciona un método de entrenamiento válido."
        }), 400

    if modo == "secuencial" and not SCRIPT_SECUENCIAL.exists():
        return jsonify({
            "ok": False,
            "mensaje": "No se encontró word2vec_secuencial_stream.py."
        }), 400

    corpus_validos = {item["ruta"] for item in obtener_corpus()}

    if len(rutas) != len(set(rutas)) or not all(
        ruta in corpus_validos for ruta in rutas
    ):
        return jsonify({
            "ok": False,
            "mensaje": "Uno o varios archivos seleccionados no son válidos."
        }), 400

    try:
        configuracion = {
            "corpus": [
                str((CARPETA / ruta).resolve())
                for ruta in rutas
            ],
            "dimension": int(datos.get("dimension", 100)),
            "ventana": int(datos.get("ventana", 5)),
            "epocas": int(datos.get("epocas", 20)),
            "min_count": int(datos.get("min_count", 1)),
            "trabajadores": 1,
            "algoritmo": datos.get("algoritmo", "skipgram")
        }

        if not 10 <= configuracion["dimension"] <= 500:
            raise ValueError("La dimensión debe estar entre 10 y 500.")

        if not 1 <= configuracion["ventana"] <= 20:
            raise ValueError("La ventana debe estar entre 1 y 20.")

        if not 1 <= configuracion["epocas"] <= 100:
            raise ValueError("Las épocas deben estar entre 1 y 100.")

        if not 1 <= configuracion["min_count"] <= 100:
            raise ValueError("La frecuencia mínima debe estar entre 1 y 100.")

        if configuracion["algoritmo"] not in ("skipgram", "cbow"):
            raise ValueError("Selecciona Skip-gram o CBOW.")

    except (TypeError, ValueError) as error:
        return jsonify({
            "ok": False,
            "mensaje": str(error) or "Revisa los parámetros seleccionados."
        }), 400

    with candado:
        if trabajo["activo"]:
            return jsonify({
                "ok": False,
                "mensaje": "Ya hay un entrenamiento en ejecución."
            }), 409

        trabajo["activo"] = True

    funcion_entrenamiento = (
        ejecutar_paralelo if modo == "paralelo"
        else ejecutar_secuencial
    )

    hilo = threading.Thread(
        target=funcion_entrenamiento,
        args=(configuracion,),
        daemon=True
    )
    hilo.start()

    return jsonify({
        "ok": True,
        "mensaje": (
            "Entrenamiento paralelo MPI iniciado en compute1."
            if modo == "paralelo"
            else "Entrenamiento secuencial iniciado con los parámetros seleccionados."
        )
    })


if __name__ == "__main__":
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    print("Word2Vec Lab disponible en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
