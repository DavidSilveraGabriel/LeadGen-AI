from crewai import Crew, Task, LLM, Process, Agent
from crewai_tools import ScrapeWebsiteTool
from utils import logger, load_yaml_config, save_lead, CompanyData, EmailData, UserProfile # Importar UserProfile
import os
import json
import datetime
from pydantic import ValidationError

# Configura el LLM para Gemini
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.6,
)

class ReportingAnalystAgent(Agent):
    """Agente Reporting Analyst con lógica para generar el reporte Markdown."""
    def perform_task(self, task: Task):
        report_date = task.inputs.get("report_date")
        report_timestamp = task.inputs.get("report_timestamp")

        markdown_report_content = "# Informe de Validación de Datos de Empresas\n\n" # Título en español
        markdown_report_content += "Este informe resume los resultados de validación para la información de empresas, contenido de emails y perfiles de usuario.\n\n" # Descripción en español
        markdown_report_content += f"## Fecha de Generación del Informe\n\n{report_date}\n\n---\n\n" # Encabezado fecha en español

        research_results = task.context[1].output if task.context and len(task.context) > 1 and task.context[1].output else []
        email_results = task.context[0].output if task.context and task.context[0].output else []
        user_profile_data = task.crew.config.get("inputs", {}).get("user_profile_dict", {})

        if not isinstance(research_results, list):
            research_results = [research_results] if research_results else []

        company_results = []
        if isinstance(research_results, list) and isinstance(email_results, list):
            company_results = list(zip(research_results, email_results))
        elif isinstance(research_results, list) and not isinstance(email_results, list):
            company_results = [(res, email_results) for res in research_results]
        elif not isinstance(research_results, list) and isinstance(email_results, list):
            company_results = [(research_results, email) for email in email_results]
        else:
            company_results = [(research_results, email_results)] if research_results or email_results else []


        for research_result, email_result in company_results:
            company_status = "success"
            company_data_section = ""
            email_content_section = ""
            user_info_section = ""
            validation_errors = None
            validated_company_data = None # Inicializar fuera del try

            try:
                if research_result:
                    validated_company_data = CompanyData(**research_result)
                    company_data_dict = validated_company_data.model_dump()

                    company_data_section = "### Información de la Empresa\n\n" # Encabezado en español
                    company_data_section += f"*   **Nombre:** {company_data_dict.get('company_name', 'N/A')}\n"
                    company_data_section += f"*   **Industria:** {company_data_dict.get('industry', 'N/A')}\n"
                    company_data_section += f"*   **Descripción:** {company_data_dict.get('about', 'N/A')}\n"

                if email_result:
                    validated_email_data = EmailData(**email_result)
                    email_data_dict = validated_email_data.model_dump()

                    email_content_section = "### Contenido del Email\n\n" # Encabezado en español
                    email_content_section += f"*   **Asunto:** {email_data_dict.get('email_subject', 'N/A')}\n"
                    email_content_section += "*   **Cuerpo:**\n    ```text\n" + email_data_dict.get('email_body', 'N/A') + "\n    ```\n"

                validated_user_profile = UserProfile(**user_profile_data)
                user_profile_dict = validated_user_profile.model_dump()

                user_info_section = "### Información del Usuario\n\n" # Encabezado en español
                user_info_section += f"*   **Nombre:** {user_profile_dict.get('name', 'N/A')}\n"
                user_info_section += f"*   **Cargo:** {user_profile_dict.get('role', 'N/A')}\n"
                user_info_section += f"*   **Información de Contacto:** {user_profile_dict.get('email', 'N/A')}\n"
                user_info_section += f"*   **Especialización:** {', '.join(user_profile_dict.get('keywords', []))}\n"

            except ValidationError as e:
                company_status = "validation_error"
                validation_errors = str(e)
                company_data_section = ""
                email_content_section = ""
                user_info_section = ""

            company_name = validated_company_data.company_name if validated_company_data and validated_company_data.company_name else 'Empresa' # Nombre de empresa en español
            markdown_report_content += f"## {company_name}\n\n" # Encabezado empresa en español
            markdown_report_content += f"**Estado:** {company_status}\n\n" # Estado en español
            markdown_report_content += f"**Timestamp:** {report_timestamp}\n\n"
            markdown_report_content += "**Datos:**\n\n" # Datos en español

            if company_status == "success":
                markdown_report_content += company_data_section
                markdown_report_content += email_content_section
                markdown_report_content += user_info_section
            elif company_status == "validation_error":
                markdown_report_content += f"**Errores de Validación:** {validation_errors}\n\n" # Errores en español

            markdown_report_content += "---\n\n"

        output_file_path = task.output_file

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_report_content)
            logger.info(f"Reporte generado y guardado en: {output_file_path}") # Log en español
            return "Reporte generado exitosamente" # Mensaje en español
        except Exception as e:
            logger.error(f"Error al guardar el reporte en Markdown: {e}", exc_info=True) # Log error en español
            return f"Error al generar el reporte: {e}" # Mensaje error en español


    def run(self, task: Task, context):
        return self.perform_task(task)


class LeadGenerationCrew:
    """Crew para la generación de leads."""

    agents_config_path = "config/agents.yaml"
    tasks_config_path = "config/tasks.yaml"

    def __init__(self, config_agents=None, config_tasks=None):
        self.agents_config = config_agents or load_yaml_config(self.agents_config_path)
        self.tasks_config = config_tasks or load_yaml_config(self.tasks_config_path)
        if self.agents_config is None or self.tasks_config is None:
            raise ValueError("No se pudieron cargar las configuraciones YAML.")

        # Inicialización diferida de agentes y tareas
        self._business_researcher = None
        self._sales_copywriter = None
        self._reporting_analyst = None

        self._research_business_task = None
        self._create_sales_email_task = None
        self._create_report_task = None

        self.crew = self._create_crew()

    def _create_crew(self):
      return Crew(
          agents=self.agents,
          tasks=self.tasks,
          process=Process.sequential,
          verbose=True
        )


    @property
    def business_researcher(self):
        if self._business_researcher is None:
            self._business_researcher = Agent(config=self.agents_config["researcher"], tools=[ScrapeWebsiteTool()], llm=gemini_llm, verbose=True, allow_delegation=False, max_iter=7, memory=True)
        return self._business_researcher

    @property
    def sales_copywriter(self):
        if self._sales_copywriter is None:
            self._sales_copywriter = Agent(config=self.agents_config["sales_copywriter"], llm=gemini_llm, verbose=True, allow_delegation=False)
        return self._sales_copywriter

    @property
    def reporting_analyst(self):
        if self._reporting_analyst is None:
            self._reporting_analyst = ReportingAnalystAgent(config=self.agents_config["reporting_analyst"], llm=gemini_llm, verbose=True, allow_delegation=False) # Usar la clase ReportingAnalystAgent
        return self._reporting_analyst


    @property
    def research_business_task(self):
        if self._research_business_task is None:
            task_config = self.tasks_config["research_business_task"]
            self._research_business_task = Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=self.business_researcher
            )
        return self._research_business_task

    @property
    def create_sales_email_task(self):
        if self._create_sales_email_task is None:
            task_config = self.tasks_config["create_sales_email_task"]
            self._create_sales_email_task = Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=self.sales_copywriter,
                context=[self.research_business_task]
            )
        return self._create_sales_email_task

    @property
    def create_report_task(self):
        if self._create_report_task is None:
            task_config = self.tasks_config["create_report_task"]
            self._create_report_task = Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=self.reporting_analyst,
                context=[self.create_sales_email_task, self.research_business_task], # Asegúrate de que research_business_task esté en context
                output_file=task_config['output_file'],
                inputs={"report_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "report_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            )
        return self._create_report_task

    @property
    def agents(self):
      return [self.business_researcher, self.sales_copywriter, self.reporting_analyst]

    @property
    def tasks(self):
      return [self.research_business_task, self.create_sales_email_task, self.create_report_task]


    def run(self, inputs):
        logger.info("Iniciando LeadGenerationCrew.run con inputs: %s", inputs)

        # Ejecuta la crew
        try:
            results = self.crew.kickoff(inputs=inputs)
            logger.info("Crew completada. Resultados: %s", results)
            return results
        except Exception as e:
            logger.exception("Error durante la ejecución de la crew:")
            return None