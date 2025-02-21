#app.py

import streamlit as st
from crewai import LLM
from crew import LeadGenerationCrew
from utils import (
    load_environment_variables,
    save_profile_data,
    load_profile_data,
    UserProfile,
    logger,
)
from typing import Dict, Any, List, Optional
import os
from pydantic import ValidationError

# --- Cargar Variables de Entorno ---
load_environment_variables()

# --- Configuración de Streamlit ---
st.set_page_config(page_title="LeadGen AI", page_icon="🚀", layout="wide")

# --- Configuración de LLM (Gemini) ---
gemini_llm = LLM(
     model="gemini/gemini-pro",
     api_key=os.environ.get("GEMINI_API_KEY"),
     temperature=0.7,
 )

# --- Funciones Auxiliares ---
def run_crewai(input_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Ejecuta el flujo de trabajo de CrewAI, manejando errores."""

    profile_data = load_profile_data()
    if profile_data is None:
        st.error("Error: Debes cargar tu perfil primero.")
        return None

    try:
        validated_user_profile = UserProfile(**profile_data)
        user_keywords = validated_user_profile.keywords or ["Problema con keywords"]
    except ValidationError as e:
        logger.error(f"Error de validación del perfil de usuario: {e}")
        st.error("Error en el perfil de usuario.  Por favor, procesa tu perfil nuevamente.")
        return None

    crew_input_data = {**input_data}
    crew_input_data["user_keywords"] = ", ".join(user_keywords)


    user_profile_dict = validated_user_profile.model_dump()
    if user_profile_dict.get("website"):
        user_profile_dict["website"] = str(user_profile_dict["website"])

    try:
        crew_instance = LeadGenerationCrew()
        # Combinar input_data y el perfil
        results = crew_instance.run(inputs={**crew_input_data, **user_profile_dict})

        if results:
            if isinstance(results, list):
                st.success("Búsqueda completada.")
                for result in results:
                    st.subheader("Datos de la Empresa")
                    st.json(result)
            elif isinstance(results, dict) and results.get("status") == "success":
                st.success("Búsqueda completada.")
                st.subheader("Datos de la Empresa")
                st.json(results["data"]["company_info"])
                st.subheader("Borrador de Correo")
                st.json(results["data"]["email_content"])
                st.subheader("Datos del Usuario")
                st.json(results["data"]["user_info"])
            elif isinstance(results, dict) and results.get("status") == "validation_error":
                st.error(f"Error de validación: {results['errors']}")
            else:
                st.error("La búsqueda no devolvió resultados o falló.")
        else:
            st.error("La búsqueda no devolvió resultados o falló.")
        return None

    except Exception as e:
        logger.error(f"Error en run_crewai: {e}", exc_info=True)
        st.error(f"Error al ejecutar CrewAI: {e}")
        return None


# --- Interfaz de Streamlit ---

st.title("🤖 LeadGen AI: Generador de Leads con IA")

# --- Barra Lateral (Sidebar) - Formulario Estructurado ---
with st.sidebar:
    st.header("Configuración")
    st.subheader("Mi Perfil")

    # Formulario con campos específicos
    with st.form("profile_form"):
        st.write("Ingresa tu información profesional:")
        name = st.text_input("Nombre Completo", placeholder="David Gabriel Silvera")
        role = st.text_input("Cargo/Título", placeholder="Científico de Datos e Ingeniero de Machine Learning e IA")
        company_name = st.text_input("Nombre de la Empresa (Opcional)", placeholder="")
        website = st.text_input("Sitio Web (Opcional)", placeholder="https://silveradavid.site/")
        phone = st.text_input("Teléfono (Opcional)", placeholder="(+54) 2657 626313")
        email = st.text_input("Correo Electrónico", placeholder="ingenieria.d.s.g@hotmail.com")
        keywords = st.text_input("Palabras Clave (separadas por comas)", placeholder="Ciencia de Datos, Machine Learning, IA")
        summary = st.text_area("Resumen Profesional", placeholder="Breve descripción de tu experiencia y habilidades", height=150)
        # Agrega más campos si es necesario (ej., intereses, especialidades, etc.)

        submitted = st.form_submit_button("Guardar Perfil")

        if submitted:
            # Crear diccionario con los datos del formulario
            profile_data = {
                "name": name,
                "role": role,
                "company_name": company_name,
                "website": website if website else None,  # Para manejar campos opcionales
                "phone": phone if phone else None,
                "email": email,  # El correo electrónico ahora es obligatorio
                "keywords": [k.strip() for k in keywords.split(",")] if keywords else [],
                "summary": summary,
                "parsing_success": True  # Ya no necesitamos el LLM para parsear, así que siempre es True
            }

            # Validar y guardar
            try:
                validated_profile = UserProfile(**profile_data)
                save_profile_data(validated_profile.model_dump())
                st.success("Perfil guardado exitosamente!")
            except ValidationError as e:
                st.error(f"Error al validar los datos del perfil: {e}")


# --- Sección Principal ---
st.header("🔍 Búsqueda de Leads")

# Un solo formulario, sin tabs
st.subheader("Búsqueda de Leads")

# Campos de entrada, combinando lo mejor de ambos formularios
# Eliminamos provincia e industria, agregamos URLs
company_urls_input = st.text_area(
    "URLs de Empresas (Máximo 3, una por línea):",
    placeholder="https://www.ejemplo-empresa1.com\nhttps://www.ejemplo-empresa2.com\nhttps://www.ejemplo-empresa3.com",
    height=100,
    key="company_urls_input"
)

lead_keywords = st.text_input("Palabras Clave Adicionales del Lead (Opcional, separadas por comas):", key="lead_keywords")


# Botón de búsqueda
if st.button("Buscar Leads"):
    # Procesar las URLs ingresadas por el usuario
    company_urls = [url.strip() for url in company_urls_input.strip().split('\n') if url.strip()]

    # Validar que no se excedan 3 URLs (opcional, pero buena práctica)
    if len(company_urls) > 3:
        st.error("Por favor, ingresa un máximo de 3 URLs.")
    elif not company_urls: # Validar que al menos haya una URL
        st.error("Por favor, ingresa al menos una URL.")
    else:
        input_data = {"company_urls": company_urls} # Pasamos las URLs como 'company_urls'
        if lead_keywords:
          input_data["keywords"] = [k.strip() for k in lead_keywords.split(",")]


        with st.spinner("Buscando leads..."):
            logger.info(f"Iniciando búsqueda con URLs: {company_urls}")
            results = run_crewai(input_data)


st.markdown("---")
st.markdown("Desarrollado por David Silvera")