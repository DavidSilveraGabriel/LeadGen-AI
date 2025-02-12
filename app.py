import streamlit as st
from crewai import LLM

# from agents import InformationExtractorAgent  # Se elimina: No se usa directamente
from crew import LeadGenerationCrew  # Se importa la Crew
from utils import (
    load_environment_variables,
    save_profile_data,
    load_profile_data,
    UserProfile,
    logger,
)  # Importamos logger desde utils
# from crewai_tools import SerperDevTool, ScrapeWebsiteTool  # No se importan aquí
from typing import Dict, Any, List, Optional
import os
import time
import logging
from pydantic import ValidationError

# --- Cargar Variables de Entorno ---
load_environment_variables()

# --- Configuración de Streamlit ---
st.set_page_config(page_title="LeadGen AI", page_icon="🚀", layout="wide")


# --- Funciones Auxiliares ---
def run_crewai(
    input_data: Dict[str, Any], search_type: str
) -> Optional[List[Dict[str, Any]]]:
    """Ejecuta el flujo de trabajo de CrewAI, manejando errores."""

    # Unifica el uso de user_keywords
    profile_data = load_profile_data()
    if profile_data is None:
        st.error("Error: Debes cargar tu perfil primero.")
        return None
    try:
        validated_user_profile = UserProfile(**profile_data)
        user_keywords = validated_user_profile.keywords or ["Problema con keywords"]  # Valor por defecto
    except ValidationError as e:
        logger.error(f"Error de validación del perfil de usuario: {e}")
        st.error("Error en el perfil de usuario.  Por favor, procesa tu perfil nuevamente.")
        return None

    # Prepara los datos para la crew, asegurando que 'user_keywords' esté presente
    crew_input_data = {**input_data}  # Copia para no modificar el original
    if search_type == "automated":
      crew_input_data["user_keywords"] = ", ".join(user_keywords)
    elif search_type == "manual":
      if "keywords" not in crew_input_data:  # Solo añade si no se proporcionaron manualmente
        crew_input_data["user_keywords"] = ", ".join(user_keywords)
    if "province" not in crew_input_data:
        crew_input_data["province"] = "Argentina"  # Valor por defecto
    if "industry" not in crew_input_data:
        crew_input_data["industry"] = "Empresas en general"  # Valor por defecto

    # Asegura que website sea string
    user_profile_dict = validated_user_profile.model_dump()
    if user_profile_dict.get("website"):
      user_profile_dict["website"] = str(user_profile_dict["website"])


    try:
        crew_instance = LeadGenerationCrew()
        crew = crew_instance.crew()

        # Combinar input_data y el perfil, usando la copia modificada
        results = crew.kickoff(inputs={**crew_input_data, **user_profile_dict})

        # Procesa y muestra los resultados (sin cambios)
        if results:
            if isinstance(results, list):  # Si es una lista, itera
                st.success("Búsqueda completada.")
                for result in results:
                    st.subheader("Datos de la Empresa")
                    st.json(
                        result
                    )  # Solo mostramos el json, se simplifica la logica
            elif (
                isinstance(results, dict) and results.get("status") == "success"
            ):  # Si es un diccionario exitoso
                st.success("Búsqueda completada.")
                st.subheader("Datos de la Empresa")
                st.json(results["data"]["company_info"])  # Accede a company_info
                st.subheader("Borrador de Correo")
                st.json(results["data"]["email_content"])  # Accede a email_content
                st.subheader("Datos del Usuario")
                st.json(results["data"]["user_info"])  # Accede a user_info
            elif (
                isinstance(results, dict)
                and results.get("status") == "validation_error"
            ):
                st.error(f"Error de validación: {results['errors']}")  # Si hay error de validacion
            else:
                st.error(
                    "La búsqueda no devolvió resultados o falló."
                )  # Otro error
        else:
            st.error("La búsqueda no devolvió resultados o falló.")  # Otro error
        return None  # Se retorna None

    except Exception as e:  # Captura genérica al final, después de los reintentos
        logger.error(f"Error en run_crewai: {e}", exc_info=True)
        st.error(f"Error al ejecutar CrewAI: {e}")
        return None


# --- Interfaz de Streamlit ---


st.title("🤖 LeadGen AI: Generador de Leads con IA")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.header("Configuración")
    st.subheader("Mi Perfil")
    profile_text = st.text_area("Pega aquí tu información en formato Markdown:", height=300)

    if st.button("Procesar Perfil"):
        if profile_text:
            logger.info("Procesando perfil...")
            result = run_crewai({"profile_text": profile_text}, search_type="profile")
            if result:
                if "error" in result[0]:
                    st.error(f"Error al procesar el perfil: {result[0]['error']}")
                elif "parsing_success" in result[0] and not result[0]["parsing_success"]:
                    st.error(
                        f"Error al procesar el perfil: {result[0].get('error', 'Error desconocido')}"
                    )
                else:
                    st.success("Perfil procesado exitosamente.")
                    st.json(result[0])
            else:
                st.error("Error al procesar el perfil (CrewAI falló).")
        else:
            st.warning("Por favor, introduce tu información de perfil.")

# --- Sección Principal ---

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

                if results and isinstance(results, list):  # Si es una lista, itera
                        st.success("Búsqueda completada.")
                        for result in results:
                            st.subheader("Datos de la Empresa")
                            st.json(
                                result
                            )  # Solo mostramos el json, se simplifica la logica
                elif (
                        isinstance(results, dict) and results.get("status") == "success"
                    ):  # Si es un diccionario exitoso
                        st.success("Búsqueda completada.")
                        st.subheader("Datos de la Empresa")
                        st.json(results["data"]["company_info"])  # Accede a company_info
                        st.subheader("Borrador de Correo")
                        st.json(results["data"]["email_content"])  # Accede a email_content
                        st.subheader("Datos del Usuario")
                        st.json(results["data"]["user_info"])  # Accede a user_info

                elif (
                    isinstance(results, dict)
                    and results.get("status") == "validation_error"
                ):
                    st.error(f"Error de validación: {results['errors']}")
                else:
                    st.error("La búsqueda no devolvió resultados o falló.")
        else:
            st.warning("Por favor, selecciona una provincia y un rubro.")


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

                if results and isinstance(results, list):  # Si es una lista, itera
                    st.success("Búsqueda completada.")
                    for result in results:
                        st.subheader("Datos de la Empresa")
                        st.json(result)  # Solo mostramos el json
                elif (
                        isinstance(results, dict) and results.get("status") == "success"
                    ):  # Si es un diccionario exitoso
                    st.success("Búsqueda completada.")
                    st.subheader("Datos")
                    st.json(results["data"]["company_info"])  # Accede a company_info
                    st.subheader("Borrador de Correo")
                    st.json(results["data"]["email_content"])  # Accede a email_content
                    st.subheader("Datos del Usuario")
                    st.json(results["data"]["user_info"])  # Accede a user_info
                elif (
                    isinstance(results, dict)
                    and results.get("status") == "validation_error"
                ):
                    st.error(f"Error de validación: {results['errors']}")
                else:
                    st.error("La búsqueda no devolvió resultados o falló.")
        else:
            st.warning("Por favor, selecciona una provincia.")

# --- Pie de Página ---
st.markdown("---")
st.markdown("Desarrollado por David Silvera")