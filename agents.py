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
    build_research_prompt,  # Se usara dentro de tasks.yaml
    build_email_prompt,
    UserProfile,
    CompanyData,
    EmailData,
    logger,
    _parse_serper_text_results, #Se importa la funcion
)

# --- Definición de Herramientas Personalizadas ---


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


class InformationExtractorAgent(Agent):
    """Agente experto en procesamiento de información."""

    def __init__(self, llm=None):  # Añade un __init__ vacío, o con llm como opcional
        super().__init__(
            role="Experto en Procesamiento de Información",
            goal="Analizar y estructurar datos de entrada, incluyendo información personal y del perfil.",
            backstory="Especialista en análisis de datos y extracción de información de perfiles.",
            verbose=True,
            allow_delegation=False,
            tools=[],
            llm=llm,
            max_iter=10,
            memory=True,
        )
        logger.info("InformationExtractorAgent inicializado.")

    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa la entrada, extrayendo información del perfil o datos de búsqueda."""
        logger.debug(f"Procesando input: {input_data}")

        if "profile_text" in input_data:
            return self._process_profile(input_data["profile_text"])
        elif all(key in input_data for key in ["industry", "province"]):
            return {
                "industry": input_data["industry"],
                "province": input_data["province"],
                "search_type": "automated",
            }
        elif any(key in input_data for key in ["company_name", "keywords"]):
            return {
                "company_name": input_data.get("company_name"),
                "keywords": input_data.get("keywords", []),
                "province": input_data.get("province"),
                "search_type": "manual",
            }
        else:
            logger.warning("Datos de entrada insuficientes para InformationExtractorAgent.")
            return {"error": "Datos de entrada insuficientes."}

    def _process_profile(self, profile_text: str) -> Dict[str, Any]:
        """Extrae información detallada del texto del perfil."""
        prompt = f"""
        Extrae la siguiente información del perfil proporcionado.  Sé EXTREMADAMENTE preciso
        con el formato.  Si un dato no está presente, devuelve 'null'.  Devuelve
        la información *sin* saltos de línea adicionales, solo texto plano en los valores.

        Texto del Perfil:
        ```
        {profile_text}
        ```

        Formato de Salida (JSON, estrictamente):
        ```json
        {{
        "name": "Nombre Completo",
        "age": "Edad (solo el número, o 'null')",
        "company_name": "Nombre de la Empresa (o 'null')",
        "role": "Título/Cargo (o 'null')",
        "website": "Sitio Web (o 'null')",
        "phone": "Número de Teléfono (o 'null')",
        "email": "string",
        "keywords": ["palabra1", "palabra2", ...],  // Lista de palabras clave, separadas por comas
        "summary": "Resumen en tono EMPRESARIAL, máximo 4 líneas, SIN saltos de línea adicionales",
        "interests": ["interes1", "interes2", ...]  // Lista de intereses, separados por comas,
        "parsing_success": true
        }}
        ```
        """
        logger.debug(f"Enviando prompt al LLM (process_profile):\n{prompt}")

        try:
            response = self.llm.call(prompt)
            logger.debug(f"Respuesta del LLM (process_profile):\n{response}")

            if (
                not response
                or response is None
                or not isinstance(response, str)
                or response.strip() == ""
            ):
                logger.error("Respuesta del LLM vacía o inválida.")
                return {
                    "error": "Respuesta del LLM vacía o inválida.",
                    "parsing_success": False,
                }

            parsed_data = self._parse_profile_response(response)
            return parsed_data

        except Exception as e:
            logger.error(f"Error al procesar el perfil con el LLM: {e}", exc_info=True)
            return {"error": str(e), "parsing_success": False}

    def _parse_profile_response(self, response: str) -> Dict[str, Any]:
        """Parsea la respuesta JSON del LLM, robusto a variaciones."""
        logger.debug(f"Parseando respuesta del LLM (perfil):\n{response}")
        result = {
            "parsing_success": False,
            "name": None,
            "age": None,
            "company_name": None,
            "role": None,
            "website": None,
            "phone": None,
            "email": None,
            "keywords": [],
            "summary": None,
            "interests": [],
        }
        try:
            response = response.strip()
            if response.startswith("\ufeff"):
                response = response.lstrip("\ufeff")

            response = response.replace("```json", "").replace("```", "").strip()

            data = json.loads(response)
            result.update(data)
            result["parsing_success"] = True

        except json.JSONDecodeError as e:
            logger.warning(
                f"JSONDecodeError: {e}.  La respuesta no es JSON válido. Respuesta:\n{response}"
            )
            result["error"] = str(e)
            return result

        except Exception as e:
            logger.error(f"Error al parsear la respuesta del perfil: {e}", exc_info=True)
            result["error"] = str(e)
            return result

        try:  # Line 192 in your snippet
            if result["age"] is not None and not isinstance(result["age"], int):
                result["age"] = int(result["age"]) if str(result["age"]).isdigit() else None
        except ValueError as e:
            logger.warning(f"ValueError during age conversion: {e}. Age set to None.")
            result["age"] = None
        except Exception as e:
            logger.error(f"Unexpected error during age validation: {e}", exc_info=True)
            result["error"] = "Error validating age"
            return result

        return result


class BusinessResearcherAgent(Agent):
    """Agente de investigación empresarial."""
    serper_tool: SerperDevTool = SerperDevTool()  # Define como atributo de clase
    scrape_tool: ScrapeWebsiteTool = ScrapeWebsiteTool()  # Define como atributo
    def __init__(self, tools: List[BaseTool] = None, llm=None):
        super().__init__(
            role="Investigador Empresarial Senior",
            goal="Obtener información detallada sobre empresas.",
            backstory="Experto en inteligencia empresarial.",
            verbose=True,
            allow_delegation=False,
            tools=tools or [],
            llm=llm,
            max_iter=15,
            memory=True,
            step_callback=self._research_logger,
        )
        logger.info("BusinessResearcherAgent inicializado.")
        #self.serper_tool = SerperDevTool() Se definen como propiedades
        #self.scrape_tool = ScrapeWebsiteTool()

    def research(
        self, search_data: Dict[str, Any], user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Investigación empresarial."""
        logger.debug("ENTRANDO a BusinessResearcherAgent.research")
        logger.debug(f"search_data: {search_data}")
        logger.debug(f"user_profile: {user_profile}")

        search_query = build_research_prompt(search_data, user_profile)
        logger.debug(f"search_query DESPUÉS de build_research_prompt: {search_query}")

        results = self._execute_research(search_query, search_data.get("province", "Argentina"))
        logger.debug("FIN de BusinessResearcherAgent.research")
        return results

    def _get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Obtiene una herramienta por su nombre."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _execute_research(self, query: str, province: str) -> List[Dict[str, Any]]:
        """Ejecuta la investigación y devuelve una lista de datos de empresas."""
        logger.debug(f"Ejecutando investigación con consulta: {query}, provincia: {province}")
        company_data_list = []
        MAX_COMPANIES = 5

        try:
            # Usa Serper
            logger.debug("Llamando a SerperDevTool...")
            # Busca la herramienta por nombre
            serper_tool = self._get_tool_by_name("SerperDevTool") #Se usa un metodo get
            if serper_tool is None:
                raise ValueError("Herramienta SerperDevTool no encontrada.")

            #Ejecuta la herramienta directamente
            serper_results = self.serper_tool.run(search_query=query) #se ejecuta la herramienta con la propiedad
            logger.debug(f"Resultados brutos de Serper: {serper_results}")

            if isinstance(serper_results, str):
                serper_results_json = _parse_serper_text_results(serper_results)
            else:
                serper_results_json = serper_results

            if "organic" in serper_results_json:
                for result in serper_results_json["organic"][:MAX_COMPANIES]:
                    try:
                        company_data = self._process_serper_result(result)
                        if not company_data:
                            continue
                        company_data["province"] = province

                        if company_data.get("website"):
                            logger.debug(f"Llamando a ScrapeWebsiteTool con URL: {company_data['website']}")
                            # Usa ScrapeWebsiteTool
                            #Busca por nombre
                            scrape_tool = self._get_tool_by_name("ScrapeWebsiteTool")
                            if scrape_tool is None:
                                raise ValueError("Herramienta ScrapeWebsiteTool no encontrada.")

                            scrape_data = self.scrape_tool.run(website_url=company_data["website"]) #Se ejecuta por la propiedad
                            company_data.update(scrape_data)
                            if isinstance(company_data.get("about"), list):
                                company_data["about"] = ". ".join(company_data["about"])

                        try:
                            validated_data = CompanyData(**company_data).model_dump()
                            company_data_list.append(validated_data)
                        except ValidationError as e:
                            logger.warning(f"Error de validación para una empresa: {e}")

                    except Exception as e:
                        logger.error(f"Error al procesar un resultado de Serper: {e}", exc_info=True)
                        continue

            return company_data_list

        except Exception as e:
            logger.error(f"Error en la investigación: {e}", exc_info=True)
            return []

    def _process_serper_result(self, result: Dict) -> Optional[Dict]:
        """Procesa un UNICO resultado de búsqueda."""
        # Extrae información relevante de un resultado individual de Serper.
        if not result.get("title") or not result.get("link"):  # Valida que tenga lo minimo
            return None
        return {
            "company_name": result.get("title"),
            "website": result.get("link"),
            "about": result.get("snippet"),
            "source": "serper",
            "fecha_consulta": datetime.now().isoformat(),
        }

    def _scrape_website(self, url: str) -> Dict:
        """Extrae información de un sitio web."""
        logger.debug(f"Extrayendo información del sitio web: {url}")
        try:
            # Busca la herramienta por nombre
            scrape_tool = self._get_tool_by_name("ScrapeWebsiteTool")
            if scrape_tool is None:
                raise ValueError("Herramienta ScrapeWebsiteTool no encontrada.")

            scraped_content = self.scrape_tool.run(website_url=url) #Se llama a la herramienta por su propiedad
            result = {
                "email": self._extract_email(scraped_content),
                "instagram": self._extract_social(scraped_content, "instagram"),
                "facebook": self._extract_social(scraped_content, "facebook"),
                "about": self._summarize_content(scraped_content),
                "source": "scrape",
            }
            logger.debug(f"Información extraída del sitio web: {result}")
            return result
        except Exception as e:
            logger.error(f"Error durante el scraping de {url}: {e}", exc_info=True)
            return {"scrape_error": str(e)}


    def _extract_email(self, text: str) -> Optional[str]:
        """Extrae una dirección de correo."""
        logger.debug(f"Extrayendo email de: {text[:100]}...")
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        result = match.group(0) if match else None
        logger.debug(f"Email encontrado: {result}")
        return result

    def _extract_social(self, text: str, platform: str) -> Optional[str]:
        """Extrae un enlace a red social."""
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