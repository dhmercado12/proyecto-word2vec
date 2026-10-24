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
    ruta = datos.get("corpus", "")

    if modo == "paralelo":
        return jsonify({
            "ok": False,
            "mensaje": "La ejecución MPI todavía no está conectada."
        }), 400

    if modo != "secuencial":
        return jsonify({
            "ok": False,
            "mensaje": "Selecciona un método de entrenamiento válido."
        }), 400

    if not SCRIPT_SECUENCIAL.exists():
        return jsonify({
            "ok": False,
            "mensaje": "No se encontró word2vec_secuencial_stream.py."
        }), 400

    corpus_validos = {item["ruta"] for item in obtener_corpus()}

    if ruta not in corpus_validos:
        return jsonify({
            "ok": False,
            "mensaje": "Selecciona un archivo TXT disponible en datos/."
        }), 400

    try:
        configuracion = {
            "corpus": str((CARPETA / ruta).resolve()),
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

    hilo = threading.Thread(
        target=ejecutar_secuencial,
        args=(configuracion,),
        daemon=True
    )
    hilo.start()

    return jsonify({
        "ok": True,
        "mensaje": "Entrenamiento secuencial iniciado con los parámetros seleccionados."
    })



if __name__ == "__main__":
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    print("Word2Vec Lab disponible en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
