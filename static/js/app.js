
const form = document.getElementById("formExperimento");
const corpusSelect = document.getElementById("corpus");
const modoSelect = document.getElementById("modo");
const botonEjecutar = document.getElementById("botonEjecutar");
const mensaje = document.getElementById("mensaje");

function primerValor(objeto, nombres, predeterminado = null) {
    for (const nombre of nombres) {
        if (objeto[nombre] !== undefined && objeto[nombre] !== null) {
            return objeto[nombre];
        }
    }
    return predeterminado;
}

function numeroBonito(valor, decimales = 2) {
    const numero = Number(valor);
    if (!Number.isFinite(numero)) return "—";
    return numero.toLocaleString("es-BO", {
        maximumFractionDigits: decimales
    });
}

function tiempoBonito(metrica) {
    const valor = primerValor(metrica, [
        "tiempo_total",
        "tiempo_total_segundos",
        "tiempo_total_s",
        "tiempo"
    ]);
    return valor === null ? "—" : `${numeroBonito(valor, 3)} s`;
}

function vocabularioBonito(metrica) {
    const valor = primerValor(metrica, [
        "vocabulario",
        "tamano_vocabulario",
        "vocab_size",
        "vocabulario_size"
    ]);
    return valor === null ? "—" : numeroBonito(valor, 0);
}

function modoBonito(metrica) {
    const nombre = String(metrica._archivo || "").toLowerCase();
    if (nombre.includes("paralelo") || nombre.includes("mpi")) {
        return "Paralelo MPI";
    }
    return "Secuencial";
}

function mostrarMetricas(metricas) {
    document.getElementById("totalExperimentos").textContent = metricas.length;

    const ultima = metricas[0];
    document.getElementById("ultimoTiempo").textContent =
        ultima ? tiempoBonito(ultima) : "—";
    document.getElementById("ultimoVocabulario").textContent =
        ultima ? vocabularioBonito(ultima) : "—";

    const cuerpo = document.getElementById("tablaMetricas");
    document.getElementById("contadorHistorial").textContent =
        `${metricas.length} ${metricas.length === 1 ? "resultado" : "resultados"}`;

    if (!metricas.length) {
        cuerpo.replaceChildren();
        const fila = document.createElement("tr");
        const celda = document.createElement("td");
        celda.colSpan = 5;
        celda.className = "empty-row";
        celda.textContent = "Todavía no hay métricas. Ejecuta tu primer experimento.";
        fila.appendChild(celda);
        cuerpo.appendChild(fila);
        return;
    }

    cuerpo.replaceChildren();

    metricas.forEach((metrica) => {
        const fila = document.createElement("tr");
        const valores = [
            metrica._archivo || "Archivo JSON",
            modoBonito(metrica),
            tiempoBonito(metrica),
            vocabularioBonito(metrica),
            metrica._modificado || "—"
        ];

        valores.forEach((valor, indice) => {
            const celda = document.createElement("td");

            if (indice === 0 || indice === 1) {
                const etiqueta = document.createElement("span");
                etiqueta.className = indice === 0 ? "file-name" : "mode-tag";
                etiqueta.textContent = valor;
                celda.appendChild(etiqueta);
            } else {
                celda.textContent = valor;
            }

            fila.appendChild(celda);
        });

        cuerpo.appendChild(fila);
    });
}

function mostrarEstado(estado) {
    document.getElementById("estadoGeneral").textContent =
        estado.estado || "Listo";
    document.getElementById("estadoProceso").textContent =
        estado.estado || "Listo";

    document.getElementById("duracionTrabajo").textContent =
        estado.activo
            ? `En ejecución · ${numeroBonito(estado.duracion_segundos, 1)} s`
            : estado.fin
                ? `Última ejecución · ${numeroBonito(estado.duracion_segundos, 1)} s`
                : "Sin ejecución activa";

    document.getElementById("detalleProceso").textContent =
        estado.error || (
            estado.activo
                ? "El entrenamiento está trabajando. Puedes seguir esta salida."
                : estado.fin
                    ? "La ejecución ha terminado."
                    : "Inicia un entrenamiento para ver su salida."
        );

    document.getElementById("salidaProceso").textContent =
        estado.salida || "Word2Vec Lab v1.0\nListo para ejecutar un experimento...";

    botonEjecutar.disabled = Boolean(estado.activo);
    botonEjecutar.textContent = estado.activo
        ? "◌ Entrenamiento en curso"
        : "▶ Iniciar entrenamiento";

    document.getElementById("orbEstado").textContent =
        estado.activo ? "…" : estado.error ? "!" : "Ⅱ";
}

async function actualizarDatos() {
    try {
        const respuesta = await fetch("/api/estado");
        const datos = await respuesta.json();

        const seleccionado = corpusSelect.value;
        corpusSelect.replaceChildren();

        if (!datos.corpus || datos.corpus.length === 0) {
            const opcion = document.createElement("option");
            opcion.value = "";
            opcion.textContent = "No se encontraron archivos TXT en datos/";
            corpusSelect.appendChild(opcion);
        } else {
            datos.corpus.forEach((archivo) => {
                const opcion = document.createElement("option");
                opcion.value = archivo.ruta;
                opcion.textContent =
                    `${archivo.nombre} (${numeroBonito(archivo.bytes / (1024 * 1024), 2)} MB)`;
                corpusSelect.appendChild(opcion);
            });

            if (datos.corpus.some((archivo) => archivo.ruta === seleccionado)) {
                corpusSelect.value = seleccionado;
            }
        }

        mostrarMetricas(datos.metricas || []);
        mostrarEstado(datos.estado || {});
    } catch (error) {
        mensaje.className = "message error";
        mensaje.textContent =
            "No se pudo conectar con Flask. Comprueba que app.py siga ejecutándose.";
    }
}

form.addEventListener("submit", async (evento) => {
    evento.preventDefault();

    mensaje.className = "message";
    mensaje.textContent = "Enviando solicitud...";
    botonEjecutar.disabled = true;

    const configuracion = {
        corpus: corpusSelect.value,
        modo: modoSelect.value,
        algoritmo: document.getElementById("algoritmo").value,
        dimension: Number(document.getElementById("dimension").value),
        ventana: Number(document.getElementById("ventana").value),
        epocas: Number(document.getElementById("epocas").value),
        min_count: Number(document.getElementById("min_count").value),
        trabajadores: 1
    };

    try {
        const respuesta = await fetch("/api/ejecutar", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(configuracion)
        });

        const datos = await respuesta.json();

        if (!respuesta.ok || !datos.ok) {
            mensaje.className = "message error";
            mensaje.textContent =
                datos.mensaje || "No se pudo iniciar el experimento.";
        } else {
            mensaje.className = "message success";
            mensaje.textContent = datos.mensaje;
        }

        await actualizarDatos();
    } catch (error) {
        mensaje.className = "message error";
        mensaje.textContent =
            "No se pudo enviar la solicitud al servidor local.";
    } finally {
        await actualizarDatos();
    }
});

document.getElementById("actualizarBtn").addEventListener(
    "click",
    actualizarDatos
);

actualizarDatos();
setInterval(actualizarDatos, 2000);
