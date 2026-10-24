# Word2Vec Lab: primera interfaz web

Esta es una primera versión local de la interfaz Flask. Está pensada para integrarse en el repositorio `proyecto-word2vec` sin reemplazar los programas de entrenamiento existentes.

## Archivos que debes copiar al repositorio

Copia `app.py`, `templates/`, `static/` y este README a la carpeta principal del proyecto. Conserva los archivos `word2vec_secuencial_stream.py`, `datos/` y `resultados/` que ya tienes.

La estructura esperada es:

```text
proyecto-word2vec/
├── app.py
├── word2vec_secuencial_stream.py
├── datos/
├── resultados/
├── templates/
│   └── index.html
└── static/
    ├── css/estilos.css
    └── js/app.js
```

## Ejecutar en Windows

Con el entorno virtual activado:

```powershell
python app.py
```

Abre en el navegador:

`http://127.0.0.1:5000`

## Qué funciona en esta versión

- Lista los archivos TXT dentro de `datos/`.
- Inicia `word2vec_secuencial_stream.py` con el corpus elegido.
- Muestra la salida del entrenamiento en la página.
- Lee los archivos `resultados/metricas_*.json` y muestra un historial básico.

## Qué falta conectar

- El entrenamiento paralelo MPI en Linux.
- Configuración de parámetros desde el formulario.
- Gráficos de comparación, speedup y eficiencia.
- Control de acceso si se expone a una red.

La opción MPI aparece como pendiente de conectar; la aplicación no afirma que el programa paralelo ya se esté ejecutando.
