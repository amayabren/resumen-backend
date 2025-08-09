# Resumen IA con Gemini, FastAPI y React

Este proyecto es una aplicación web que permite resumir textos usando la API gratuita de Google Gemini, guardar los resultados en una base de datos SQLite y exportarlos en PDF.
El usuario puede editar el prompt, pegar cualquier texto y obtener un resumen claro y estructurado.

## Tecnologías utilizadas
### Backend (Python + FastAPI)
FastAPI → Framework rápido para crear APIs.

SQLite → Base de datos local y ligera para guardar resúmenes.

SQLAlchemy → ORM para manejar la base de datos de forma sencilla.

google-generativeai → Cliente oficial de Google Gemini para Python.

ReportLab → Generador de PDFs en Python.

dotenv → Para cargar la API key desde un archivo .env.

### Frontend (JavaScript + React)
React → Librería para la interfaz de usuario.

Vite → Herramienta de desarrollo rápida para React.

TailwindCSS → Framework de estilos.

Axios → Cliente HTTP para comunicarse con el backend.


