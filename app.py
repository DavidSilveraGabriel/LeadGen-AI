import streamlit as st
from crewai import LLM
from crew import LeadGenerationCrew
# from agents import InformationExtractorAgent  # Ya no se necesita
from utils import (
    load_environment_variables,
    save_profile_data,
    load_profile_data,
    UserProfile,
    logger,
)
from typing import Dict, Any, List, Optional
import os
# import time  # Ya no se necesita la función retry en app.py
# import logging  # Ya no se necesita logging directamente en app.py
from pydantic import ValidationError

# --- Cargar Variables de Entorno ---
load_environment_variables()

# --- Configuración de Streamlit ---
st.set_page_config(page_title="LeadGen AI", page_icon="🚀", layout="wide")


# --- Configuración de LLM (Gemini) ---
# (Mantén tu configuración de Gemini como la tenías, ya que está correcta)
from crewai import LLM
import os

gemini_llm = LLM(
     model="gemini/gemini-pro",
     api_key=os.environ.get("GEMINI_API_KEY"),
     temperature=0.7,
 )

# --- Funciones Auxiliares ---
def run_crewai(
    input_data: Dict[str, Any], search_type: str
) -> Optional[List[Dict[str, Any]]]:
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
    if search_type == "automated":
        crew_input_data["user_keywords"] = ", ".join(user_keywords)
    elif search_type == "manual":
        if "keywords" not in crew_input_data:
            crew_input_data["user_keywords"] = ", ".join(user_keywords)
    if "province" not in crew_input_data:
        crew_input_data["province"] = "Argentina"
    if "industry" not in crew_input_data:
        crew_input_data["industry"] = "Empresas en general"

    user_profile_dict = validated_user_profile.model_dump()
    if user_profile_dict.get("website"):
        user_profile_dict["website"] = str(user_profile_dict["website"])

    try:
        crew_instance = LeadGenerationCrew()
        # Combinar input_data y el perfil
        results = crew_instance.run(inputs={**crew_input_data, **user_profile_dict}) #Se llama al metodo run de crew

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


# --- Sección Principal (resto de la app, sin cambios mayores) ---

st.header("🔍 Búsqueda de Leads")
tab_auto, tab_manual = st.tabs(["Búsqueda Automatizada", "Búsqueda Manual"])

with tab_auto:
    st.subheader("Búsqueda Automatizada")
    province_auto = st.selectbox(
        "Provincia:",
        ["Buenos Aires", "Córdoba", "Santa Fe", "Mendoza", "Tucumán", "Otra"],
    )
    if province_auto == "Otra":
        province_auto = st.text_input("Escribe la provincia:")
    industry_auto = st.selectbox(
        "Rubro:",
        [
            "Agencias de Marketing Digital",
            "Tiendas Online (e-commerce)",
            "Empresas de Software/SaaS",
            "Consultoras",
            "Startups",
            "Servicios Financieros (pequeños)",
            "Educación",
            "Salud",
            "Servicios Profesionales",
            "Otro",
        ],
    )
    if industry_auto == "Otro":
        industry_auto = st.text_input("Escribe el rubro:")

    if st.button("Buscar Leads (Automatizado)"):
        if province_auto and industry_auto:
            with st.spinner("Buscando leads..."):
                logger.info(
                    f"Iniciando búsqueda automatizada (provincia: {province_auto}, rubro: {industry_auto})..."
                )
                results = run_crewai(
                    {"province": province_auto, "industry": industry_auto},
                    search_type="automated",
                )
                # El resto del código para mostrar resultados se mantiene igual

with tab_manual:
    st.subheader("Búsqueda Manual")
    company_name_manual = st.text_input("Nombre de la Empresa (Opcional):")
    keywords_manual = st.text_input("Palabras Clave (Opcional):")
    province_manual = st.selectbox(
        "Provincia:",
        ["Buenos Aires", "Córdoba", "Santa Fe", "Mendoza", "Tucumán", "Otra"],
        key="province_manual",
    )
    if province_manual == "Otra":
        province_manual = st.text_input("Escribe la provincia:", key="province_manual_text")

    if st.button("Buscar Leads (Manual)"):
        if province_manual:
            input_data = {"province": province_manual}
            if company_name_manual:
                input_data["company_name"] = company_name_manual
            if keywords_manual:
                input_data["keywords"] = [k.strip() for k in keywords_manual.split(",")]

            with st.spinner("Buscando leads..."):
                logger.info(f"Iniciando búsqueda manual: {input_data}")
                results = run_crewai(input_data, search_type="manual")
# El resto del código para mostrar resultados se mantiene igual

st.markdown("---")
st.markdown("Desarrollado por David Silvera")