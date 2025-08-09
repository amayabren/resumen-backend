from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import google.generativeai as genai
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime
import os
import re

# ------------------ Configuración ------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Base de datos ------------------
DATABASE_URL = "sqlite:///./resumenes.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

class Resumen(Base):
    __tablename__ = "resumenes"
    id = Column(Integer, primary_key=True, index=True)
    creado_en = Column(DateTime, default=datetime.now)
    prompt = Column(Text)
    texto_original = Column(Text)
    respuesta = Column(Text)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ------------------ Utils ------------------
def limpiar_markdown(texto):
    """Quita negritas y cursivas pero mantiene las listas y viñetas."""
    if not texto:
        return ""
    # Quitar **negrita** y __negrita__
    texto = re.sub(r"\*\*(.*?)\*\*", r"\1", texto)
    texto = re.sub(r"__(.*?)__", r"\1", texto)
    # Quitar cursivas solas *cursiva* o _cursiva_
    texto = re.sub(r"(?<!\n)\*(?!\s)(.*?)\*(?!\s)", r"\1", texto)
    texto = re.sub(r"(?<!\w)_(?!\s)(.*?)_(?!\s)", r"\1", texto)
    return texto

# ------------------ Modelos ------------------
class TextoEntrada(BaseModel):
    prompt: str
    texto: str

# ------------------ Endpoints ------------------
@app.post("/resumir")
async def resumir_texto(data: TextoEntrada):
    try:
        prompt_completo = f"{data.prompt}\n\n{data.texto}"

        # Usar modelo recomendado
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt_completo)

        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("La respuesta de Gemini está vacía o incompleta.")

        resumen_texto = limpiar_markdown(response.candidates[0].content.parts[0].text)

        # Guardar en DB
        db = SessionLocal()
        nuevo_resumen = Resumen(
            prompt=data.prompt,
            texto_original=data.texto,
            respuesta=resumen_texto
        )
        db.add(nuevo_resumen)
        db.commit()
        db.refresh(nuevo_resumen)
        db.close()

        return {
            "id": nuevo_resumen.id,
            "resumen": resumen_texto
        }

    except Exception as e:
        print("ERROR en /resumir:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/exportar/{id}")
async def exportar_pdf(id: int):
    db = SessionLocal()
    resumen_obj = db.query(Resumen).filter(Resumen.id == id).first()
    db.close()

    if not resumen_obj:
        raise HTTPException(status_code=404, detail="Resumen no encontrado")

    file_path = f"resumen_{id}.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    flow = []

    # Encabezado
    flow.append(Paragraph("Resumen IA", styles["Title"]))
    flow.append(Paragraph(f"ID: {resumen_obj.id}", styles["Normal"]))
    flow.append(Paragraph(f"Creado: {resumen_obj.creado_en}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    # Prompt
    flow.append(Paragraph("Prompt:", styles["Heading2"]))
    flow.append(Paragraph(limpiar_markdown(resumen_obj.prompt), styles["Normal"]))
    flow.append(Spacer(1, 12))

    # Texto Original
    flow.append(Paragraph("Texto Original:", styles["Heading2"]))
    flow.append(Paragraph(limpiar_markdown(resumen_obj.texto_original), styles["Normal"]))
    flow.append(Spacer(1, 12))

    # Respuesta IA (respetar saltos de línea)
    flow.append(Paragraph("Respuesta IA:", styles["Heading2"]))
    texto_respuesta = limpiar_markdown(resumen_obj.respuesta)
    texto_respuesta = texto_respuesta.replace("\n", "<br/>")  # Forzar saltos
    flow.append(Paragraph(texto_respuesta, styles["Normal"]))


    doc.build(flow)

    return FileResponse(file_path, filename=f"resumen_{id}.pdf", media_type="application/pdf")
