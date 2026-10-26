

const form = document.getElementById("formExperimento");
const corpusSelect = document.getElementById("corpus");
const opcionesCorpus = document.getElementById("opcionesCorpus");
const textoCorpus = document.getElementById("textoCorpus");
const selectorMultiple = document.getElementById("selectorMultiple");
const modoSelect = document.getElementById("modo");

const botonEjecutar = document.getElementById("botonEjecutar");
const mensaje = document.getElementById("mensaje");


function obtenerRutasSeleccionadas() {
    return Array.from(
        opcionesCorpus.querySelectorAll('input[type="checkbox"]:checked')
    ).map(checkbox => checkbox.value);
}

function actualizarTextoCorpus() {
    const seleccionados = opcionesCorpus.querySelectorAll(
        'input[type="checkbox"]:checked'
    );

    if (seleccionados.length === 0) {
        textoCorpus.textContent = "Selecciona uno o varios archivos";
    } else if (seleccionados.length === 1) {
        textoCorpus.textContent = seleccionados[0].dataset.nombre;
    } else {
        textoCorpus.textContent = `${seleccionados.length} archivos seleccionados`;
    }
}

function cargarCorpus(archivos) {
    const seleccionAnterior = new Set(obtenerRutasSeleccionadas());

    opcionesCorpus.replaceChildren();

    if (!archivos || archivos.length === 0) {
        textoCorpus.textContent = "No hay archivos TXT disponibles";
        corpusSelect.disabled = true;
        return;
    }

    corpusSelect.disabled = false;

    archivos.forEach(archivo => {
        const etiqueta = document.createElement("label");
        const casilla = document.createElement("input");
        const nombre = document.createElement("span");

        casilla.type = "checkbox";
        casilla.value = archivo.ruta;
        casilla.dataset.nombre = archivo.nombre;
        casilla.checked = seleccionAnterior.has(archivo.ruta);

        nombre.textContent =
            `${archivo.nombre} (${numeroBonito(archivo.bytes / (1024 * 1024), 2)} MB)`;

        casilla.addEventListener("change", actualizarTextoCorpus);

        etiqueta.append(casilla, nombre);
        opcionesCorpus.appendChild(etiqueta);
    });

    actualizarTextoCorpus();
}

corpusSelect.addEventListener("click", () => {
    const estaAbierto = !opcionesCorpus.hidden;
    opcionesCorpus.hidden = estaAbierto;
    corpusSelect.setAttribute("aria-expanded", String(!estaAbierto));
});

document.addEventListener("click", evento => {
    if (!selectorMultiple.contains(evento.target)) {
        opcionesCorpus.hidden = true;
        corpusSelect.setAttribute("aria-expanded", "false");
    }
});


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
    const ultima = metricas[0];

    document.getElementById("resultadoTiempo").textContent =
        ultima ? tiempoBonito(ultima) : "—";

    document.getElementById("resultadoVocabulario").textContent =
        ultima ? vocabularioBonito(ultima) : "—";

    document.getElementById("resultadoDimension").textContent =
        ultima
            ? numeroBonito(primerValor(ultima, [
                "dimension",
                "vector_size",
                "dimensiones"
            ]), 0)
            : "—";

    document.getElementById("resultadoPalabras").textContent =
        ultima
            ? numeroBonito(primerValor(ultima, [
                "palabras_procesadas",
                "palabras",
                "words",
                "total_palabras"
            ]), 0)
            : "—";

    const cuerpo = document.getElementById("tablaMetricas");

    document.getElementById("contadorHistorial").textContent =
        `${metricas.length} ${
            metricas.length === 1 ? "resultado" : "resultados"
        }`;

    cuerpo.replaceChildren();

    if (!metricas.length) {
        const fila = document.createElement("tr");
        const celda = document.createElement("td");

        celda.colSpan = 5;
        celda.className = "empty-row";
        celda.textContent =
            "Todavía no hay métricas. Ejecuta tu primer experimento.";

        fila.appendChild(celda);
        cuerpo.appendChild(fila);
        return;
    }

    metricas.forEach((metrica) => {
        const fila = document.createElement("tr");

        const valores = [
            metrica._archivo || "Archivo JSON",
            modoBonito(metrica),
            tiempoBonito(metrica),
            vocabularioBonito(metrica),
            metrica._modificado || "—"
        ];

        valores.forEach((valor) => {
            const celda = document.createElement("td");
            celda.textContent = valor;
            fila.appendChild(celda);
        });

        cuerpo.appendChild(fila);
    });
}



function mostrarComparacion(metricas) {
    const tiempoSecuencial = document.getElementById("tiempoSecuencial");
    const tiempoParalelo = document.getElementById("tiempoParalelo");
    const resultado = document.getElementById("resultadoComparacion");

    if (!tiempoSecuencial || !tiempoParalelo || !resultado) return;

    const secuenciales = metricas.filter(
        m => modoBonito(m) === "Secuencial"
    );

    const paralelos = metricas.filter(
        m => modoBonito(m) === "Paralelo MPI"
    );

    const obtenerTiempo = m => {
        const valor = primerValor(m, [
            "tiempo_total",
            "tiempo_total_segundos",
            "tiempo_total_s",
            "tiempo"
        ]);

        return valor === null ? NaN : Number(valor);
    };

    const secuencial = secuenciales[0];
    const paralelo = paralelos[0];

    if (!secuencial || !paralelo) {
        tiempoSecuencial.textContent = secuencial
            ? numeroBonito(obtenerTiempo(secuencial), 3)
            : "—";

        tiempoParalelo.textContent = paralelo
            ? numeroBonito(obtenerTiempo(paralelo), 3)
            : "—";

        resultado.innerHTML = "";
        const titulo = document.createElement("h3");
        const detalle = document.createElement("p");

        titulo.textContent = "Comparación pendiente";
        detalle.textContent =
            "Ejecuta ambos métodos para ver cuál tarda menos.";

        resultado.append(titulo, detalle);
        return;
    }

    const ts = obtenerTiempo(secuencial);
    const tp = obtenerTiempo(paralelo);

    tiempoSecuencial.textContent = numeroBonito(ts, 3);
    tiempoParalelo.textContent = numeroBonito(tp, 3);

    const diferencia = Math.abs(ts - tp);

    let titulo;
    let detalle;

    if (!Number.isFinite(ts) || !Number.isFinite(tp)) {
        titulo = "No se pudo realizar la comparación";
        detalle = "Revisa los tiempos guardados en los archivos de métricas.";
    } else if (ts < tp) {
        titulo = "El secuencial fue más rápido";
        detalle =
            `Terminó ${numeroBonito(diferencia, 3)} segundos antes que MPI.`;
    } else if (tp < ts) {
        titulo = "¡MPI fue más rápido!";
        detalle =
            `Terminó ${numeroBonito(diferencia, 3)} segundos antes que el secuencial.`;
    } else {
        titulo = "Empate en tiempo";
        detalle = "Ambos métodos registraron el mismo tiempo de ejecución.";
    }

    resultado.replaceChildren();

    const encabezado = document.createElement("h3");
    const descripcion = document.createElement("p");

    encabezado.textContent = titulo;
    descripcion.textContent = detalle;

    resultado.append(encabezado, descripcion);
}



function mostrarEstado(estado) {
    const estadoProceso = document.getElementById("estadoProceso");
    const duracionTrabajo = document.getElementById("duracionTrabajo");
    const detalleProceso = document.getElementById("detalleProceso");
    const salidaProceso = document.getElementById("salidaProceso");
    const orbEstado = document.getElementById("orbEstado");

    if (estadoProceso) {
        estadoProceso.textContent = estado.estado || "Listo";
    }

    if (duracionTrabajo) {
        duracionTrabajo.textContent = estado.activo
            ? `En ejecución · ${numeroBonito(estado.duracion_segundos, 1)} s`
            : estado.fin
                ? `Última ejecución · ${numeroBonito(estado.duracion_segundos, 1)} s`
                : "Sin ejecución activa";
    }

    if (detalleProceso) {
        detalleProceso.textContent = estado.error || (
            estado.activo
                ? "El entrenamiento está trabajando. Puedes seguir esta salida."
                : estado.fin
                    ? "La ejecución ha terminado."
                    : "Inicia un entrenamiento para ver su salida."
        );
    }

    if (salidaProceso) {
        salidaProceso.textContent = estado.salida ||
            "Word2Vec Lab v1.0\nListo para ejecutar un experimento...";
    }

    botonEjecutar.disabled = Boolean(estado.activo);
    botonEjecutar.textContent = estado.activo
        ? "◌ Entrenamiento en curso"
        : "▶ Iniciar entrenamiento";

    if (orbEstado) {
        orbEstado.textContent = estado.activo
            ? "…"
            : estado.error
                ? "!"
                : "Ⅱ";
    }
}




async function actualizarDatos() {
    try {
        const respuesta = await fetch("/api/estado");
        const datos = await respuesta.json();

        cargarCorpus(datos.corpus || []);

        const metricas = datos.metricas || [];
        mostrarMetricas(metricas);
        mostrarComparacion(metricas);

        mostrarEstado(datos.estado || {});
    } catch (error) {
        mensaje.className = "message error";
        mensaje.textContent =
            "No se pudo conectar con Flask. Comprueba que app.py siga ejecutándose.";
    }
}



form.addEventListener("submit", async (evento) => {
    evento.preventDefault();

    if (obtenerRutasSeleccionadas().length === 0) {
    mensaje.className = "message error";
    mensaje.textContent = "Selecciona al menos un archivo TXT.";
    return;
}

    mensaje.className = "message";
    mensaje.textContent = "Enviando solicitud...";
    botonEjecutar.disabled = true;

    const configuracion = {
        corpus: obtenerRutasSeleccionadas(),
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
