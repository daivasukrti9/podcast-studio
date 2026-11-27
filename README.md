# Podcast Studio

Estudio local de generación y gestión de podcasts en Python y FastAPI.

## Características

- Interfaz dual: Modo Estudio (diseño vidrio diurno) y Modo Librería (estilo reproductor con playlists).
- Soporte multilingüe para emisión en Español, Inglés y Portugués con 24 voces neuronales integradas.
- Ingesta de fuentes desde URLs y documentos (.md, .txt, .docx).
- Traducción automática de contenidos al idioma de destino seleccionado.
- Motor de guiones con perfiles conversacionales y síntesis de voz en tomas continuas.
- Cadena de masterización de audio determinista (DSP sin IA, filtros pasa-altos, compresor y auto-ducking).
- Bóveda de autenticación y llaves API descentralizadas para colaboradores.
- Catálogo de 16 programas temáticos con portadas predefinidas.

## Requisitos

- Python 3.10 o superior.
- FFmpeg (incluido o referenciado mediante imageio-ffmpeg).

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

En Windows:
Ejecutar `iniciar_estudio.bat` o:

```bash
python app.py
```

Acceder desde el navegador a: `http://localhost:8085`
