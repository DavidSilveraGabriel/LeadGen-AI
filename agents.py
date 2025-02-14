# agents.py (MODIFICADO - Usando CrewBase y YAML)
from datetime import datetime
import json
import re
from typing import List, Dict, Any, Optional

from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.tools import BaseTool
from pydantic import ValidationError


from utils import (
    save_lead,
    check_lead_exists,
    search_lead,
    build_research_prompt,
    build_email_prompt,
    UserProfile,
    CompanyData,
    EmailData,
    logger,
)

# --- Definición de Herramientas Personalizadas ---
# (Estas herramientas no se usan directamente, pero las dejo por si las necesitas en el futuro)
class CheckLeadExistsTool(BaseTool):
    name: str = "Check Lead Exists"
    description: str = "Verifica si un lead ya existe en la base de datos."

    def _run(self, company_name: str, province: str, **kwargs) -> bool:
        logger.debug(
            f"Verificando si el lead '{company_name}' en '{province}' existe..."
        )
        exists = check_lead_exists(company_name, province)
        logger.debug(f"Resultado de la verificación: {exists}")
        return exists


class SearchLeadTool(BaseTool):
    name: str = "Search Lead"
    description: str = "Busca y recupera información de un lead existente."

    def _run(
        self, company_name: str, province: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        logger.debug(f"Buscando lead: '{company_name}' en '{province}'...")
        result = search_lead(company_name, province)
        logger.debug(f"Resultado de la búsqueda: {result}")
        return result


# --- Agentes ---
# information_extractor ya no se usa.

class BusinessResearcherAgent(Agent):
    """Agente de investigación empresarial."""

    # Las herramientas se pasan como argumento al inicializar
    def __init__(self, tools: List[BaseTool] = None, llm=None):
        super().__init__(
            role="Investigador Empresarial Senior",
            goal="Obtener información detallada sobre empresas.",
            backstory="Experto en inteligencia empresarial.",
            verbose=True,
            allow_delegation=False,
            tools=tools or [],  # Usa las herramientas que se le pasen (o una lista vacía)
            llm=llm,
            max_iter=7,  # Aumenta las iteraciones máximas
            memory=True,
            step_callback=self._research_logger,
        )
        logger.info("BusinessResearcherAgent inicializado.")


    def research(
        self, search_data: Dict[str, Any], user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Investigación empresarial."""
        logger.debug("ENTRANDO a BusinessResearcherAgent.research")
        logger.info(f"search_data: {search_data}")
        logger.info(f"user_profile: {user_profile}")

        search_query = build_research_prompt(search_data, user_profile)
        
        logger.debug(f"search_query DESPUÉS de build_research_prompt: {search_query}")

        results = self._execute_research(search_query, search_data.get("province", "Argentina"))
        logger.info(f"Resultados de la investigación: {results}")
        logger.debug("FIN de BusinessResearcherAgent.research")
        return results


    def _execute_research(self, query: str, province: str) -> List[Dict[str, Any]]:
        """Ejecuta la investigación y devuelve una lista de datos de empresas."""
        logger.debug(f"Ejecutando investigación con consulta: '{query}', provincia: '{province}'")
        company_data_list = []
        MAX_COMPANIES = 5

        try:
            # 1. Usar Serper para buscar.
            logger.debug("Llamando a SerperDevTool...")
            serper_results = self.tools[0].run(search_query=query)  # SOLUCIÓN: query es una string
            logger.debug(f"Resultados brutos de Serper: {serper_results}")

            # 2. Procesamiento de la respuesta de Serper
            if isinstance(serper_results, str):
                try:
                    serper_results_json = json.loads(serper_results)
                except json.JSONDecodeError:
                    logger.error(f"Error al decodificar JSON de Serper: {serper_results}")
                    return []
            elif isinstance(serper_results, dict):
                serper_results_json = serper_results
            else:
                logger.error(f"Tipo de respuesta inesperado de Serper: {type(serper_results)}")
                return []
            
            
            logger.debug(f"Serper results (JSON): {serper_results_json}") # NUEVO

            # 3. Extraer URLs y verificar que 'organic' exista
            urls_to_scrape = []
            if "organic" in serper_results_json:
               for result in serper_results_json["organic"][:MAX_COMPANIES]:
                  if result.get("link"):
                     urls_to_scrape.append(result["link"])
            else:
                logger.warning("La respuesta de Serper no contiene resultados orgánicos.")
                return [] #No encontró resultados


            # 4. Scrape de cada URL y procesamiento con LLM
            for url in urls_to_scrape:
                try:
                    logger.debug(f"Llamando a ScrapeWebsiteTool con URL: {url}")
                    scraped_content = self.tools[1].run(website_url=url)
                    # Limitamos el log para que no sea tan extenso
                    logger.debug(f"Contenido scrapeado (primeros 500 chars): {scraped_content[:500]}...")

                    # 5. Usar el LLM para extraer información.
                    prompt = f"""
                    De la siguiente página web, extrae la información de contacto
                    de la empresa, si es que existe.
                    Si la página web es de una empresa que ofrece servicios de marketing digital,
                    y en la página se menciona explícitamente el uso de 'ciencia de datos',
                    'inteligencia artificial', 'IA', 'GenAI', 'agentes de IA',
                    o servicios relacionados, extrae también el nombre de la empresa,
                    su sitio web, y una breve descripción de la empresa (máximo 200 caracteres).

                    Si no encuentras alguno de los datos solicitados, devuelve 'null' para ese campo.

                    Página web:
                    ```
                    {scraped_content[:6000]}  # Aumenté el límite, pero aún lo limitamos.
                    ```
                    Formato de Salida (JSON, estrictamente):
                    {{
                        "company_name": "Nombre de la Empresa",
                        "website": "URL válida del sitio web",
                        "description": "Breve descripción",
                        "email": "Correo electrónico (o null)",
                        "social_media": {{
                            "linkedin": "URL de LinkedIn (o null)",
                            "twitter": "URL de Twitter (o null)",
                            "facebook": "URL de Facebook (o null)",
                            "instagram": "URL de Instagram (o null)"
                        }}
                    }}
                    """

                    logger.debug(f"Enviando prompt al LLM para extraer datos de: {url}")
                    response = self.llm.call(prompt)
                    logger.debug(f"Respuesta del LLM: {response}")


                    if response:
                         try:
                            response_json = json.loads(response.replace("```json", "").replace("```", "").strip())
                            response_json["website"] = url  # Asegura que la URL sea la correcta
                            response_json["source"] = "scrape"
                            response_json["fecha_consulta"] = datetime.now().isoformat()
                            response_json["province"] = province

                            # Validación con Pydantic
                            validated_data = CompanyData(**response_json).model_dump()
                            company_data_list.append(validated_data)

                         except json.JSONDecodeError:
                            logger.error(f"Error al decodificar JSON del LLM: {response}")
                         except ValidationError as e:
                             logger.warning(f"Error de validación Pydantic: {e}")

                except Exception as e:
                    logger.exception(f"Error durante el scraping o procesamiento de {url}: {e}")

            return company_data_list

        except Exception as e:
            logger.exception(f"Error en la investigación: {e}")  # Captura genérica
            return []


    def _extract_email(self, text: str) -> Optional[str]:
        """Extrae una dirección de correo (función auxiliar)."""
        logger.debug(f"Extrayendo email de: {text[:100]}...")  # Log primeros 100 chars
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        result = match.group(0) if match else None
        logger.debug(f"Email encontrado: {result}")
        return result

    def _extract_social(self, text: str, platform: str) -> Optional[str]:
        """Extrae un enlace a red social (función auxiliar)."""
        logger.debug(f"Extrayendo enlace de {platform} de: {text[:100]}...")
        if platform == "instagram":
            match = re.search(r"instagram\.com/([a-zA-Z0-9_.]+)", text)
        elif platform == "facebook":
            match = re.search(r"facebook\.com/([a-zA-Z0-9_.]+)", text)
        else:
            logger.warning(f"Plataforma no soportada: {platform}")
            return None
        result = match.group(0) if match else None
        logger.debug(f"Enlace de {platform} encontrado: {result}")
        return result
    def _summarize_content(self, text: str) -> str:
        """Resumen automático usando el LLM."""
        prompt = f"Resume en 200 caracteres: {text[:3000]}"
        logger.debug(f"Enviando prompt para resumen: {prompt}")
        try:
            response = self.llm.call(prompt)
            logger.debug(f"Resumen generado: {response}")
            return response
        except Exception as e:
            logger.error(f"Error al resumir con el LLM: {e}", exc_info=True)
            return "Error al generar el resumen."

    def _research_logger(self, step_output):
        """Registro detallado de pasos de investigación."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step": step_output.description,
                "agent": self.role,
                "result": step_output.return_values,
            }
        except AttributeError:
             log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step": "Error en el paso",
                "agent": self.role,
                "result": str(step_output),
            }
        with open("research_log.jsonl", "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

class SalesCopywriterAgent(Agent):
    """Especialista en redacción de ventas."""

    def __init__(self, llm=None):
        super().__init__(
            role="Redactor Comercial IA",
            goal="Crear correos electrónicos de ventas personalizados.",
            backstory="Redactor de ventas B2B con experiencia.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=5,
            memory=True,
        )
        logger.info("SalesCopywriterAgent inicializado.")

    def write_email(self, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Escribe el correo electrónico usando el LLM y la información."""
        logger.debug(f"Escribiendo correo electrónico con contexto: {context}")

        # Extrae la información del contexto
        company_data = (
            context[0] if len(context) > 0 and isinstance(context[0], dict) else {}
        )
        user_profile = (
            context[1] if len(context) > 1 and isinstance(context[1], dict) else {}
        )

        # Construye el prompt con la función de prompting
        prompt = build_email_prompt(company_data, user_profile)
        logger.debug(f"PROMPT para email: {prompt}")  # Añade este log

        try:
            response = self.llm.call(prompt)
            logger.debug(f"Respuesta del LLM (write_email): {response}")
            structured_response = self._structure_email(response)
            logger.debug(f"Correo electrónico estructurado: {structured_response}")
            return structured_response
        except Exception as e:
            logger.error(f"Error al generar el correo electrónico: {e}", exc_info=True)
            return {"error": str(e)}

    def _structure_email(self, raw_text: str) -> Dict:
        """Estructura la respuesta del LLM en un diccionario."""
        lines = raw_text.split("\n")
        return {
            "email_subject": lines[0].replace("Asunto:", "").strip()
            if lines
            else "Propuesta",
            "email_body": "\n".join(lines[1:]).strip(),
            "keywords": self._extract_keywords(raw_text),  # Podrías mejorar esto
            "generated_at": datetime.now().isoformat(),
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extrae palabras clave del texto del correo (función básica)."""
        response = self.llm.call(f"Extrae 5 palabras clave: {text}")
        return response.split(", ")


class ReportingAnalystAgent(Agent):
    """Analista de Datos."""

    def __init__(self, llm=None):
        super().__init__(
            role="Analista de Datos Senior",
            goal="Organizar la información y guardarla.",
            backstory="Experto en business intelligence.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True,
        )
        logger.info("ReportingAnalystAgent inicializado.")

    def create_report(self, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Crea el informe final, valida y guarda los datos."""
        logger.debug(f"Creando informe con contexto: {context}")
        # Extrae datos del contexto
        company_data = context[0] if context else {}
        email_data = context[1] if len(context) > 1 else {}
        user_profile = context[2] if len(context) > 2 else {}

        # 1. VALIDA los datos (usando Pydantic)
        try:
            validated_company = CompanyData(**company_data)
            validated_email = EmailData(**email_data)
            # validated_user = UserProfile(**user_profile)  # No es necesario guardarlo
        except ValidationError as e:
            logger.error(f"Error de validación: {e}")
            return {"status": "validation_error", "errors": str(e)}

        # 2. Guarda el lead (si la validación es exitosa)
        try:
            save_lead(validated_company.model_dump())  # Guarda la info de la empresa
            logger.info(f"Lead guardado: {validated_company.company_name}")
        except Exception as e:
            logger.error(f"Error al guardar el lead: {e}", exc_info=True)
            # Aquí se usa el error de validación, NO el error genérico
            return {"status": "error", "errors": str(e)}


        # 3. Construye el informe final
        report = {
            "company_info": validated_company.model_dump(),
            "email_content": validated_email.model_dump(),
            "user_info": user_profile,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_sources": [company_data.get("source", "unknown")],
            },
        }
        return {"status": "success", "data": report, "timestamp": datetime.now().isoformat()}