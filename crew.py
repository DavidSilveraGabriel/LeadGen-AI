from crewai import Crew, Task, LLM, Process
from crewai_tools import ScrapeWebsiteTool # Solo importamos ScrapeWebsiteTool
from utils import logger, load_yaml_config, save_lead, CompanyData, EmailData # Importar para validación
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
        from crewai import Agent
        if self._business_researcher is None:
            # Solo ScrapeWebsiteTool, quitamos SerperDevTool
            self._business_researcher = Agent(config=self.agents_config["researcher"], tools=[ScrapeWebsiteTool()], llm=gemini_llm, verbose=True, allow_delegation=False, max_iter=7, memory=True)
        return self._business_researcher

    @property
    def sales_copywriter(self):
        from crewai import Agent
        if self._sales_copywriter is None:
            self._sales_copywriter = Agent(config=self.agents_config["sales_copywriter"], llm=gemini_llm, verbose=True, allow_delegation=False)
        return self._sales_copywriter

    @property
    def reporting_analyst(self):
        from crewai import Agent
        if self._reporting_analyst is None:
            self._reporting_analyst = Agent(config=self.agents_config["reporting_analyst"], llm=gemini_llm, verbose=True, allow_delegation=False)
        return self._reporting_analyst


    @property
    def research_business_task(self):
        from crewai import Task
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
        from crewai import Task
        if self._create_sales_email_task is None:
            task_config = self.tasks_config["create_sales_email_task"]
            self._create_sales_email_task = Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=self.sales_copywriter,
                context=[self.research_business_task] # <--- Contexto correcto
            )
        return self._create_sales_email_task

    @property
    def create_report_task(self):
        from crewai import Task
        if self._create_report_task is None:
            task_config = self.tasks_config["create_report_task"]
            self._create_report_task = Task(
                description=task_config['description'],
                expected_output=task_config['expected_output'],
                agent=self.reporting_analyst,
                context=[self.create_sales_email_task],
                output_file=task_config['output_file'], #Se pasa el parametro directamente a la Tarea.
                inputs={"report_date": datetime.datetime.now().strftime("%Y-%m-%d"), # <--- Fecha actual
                        "report_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} # <--- Timestamp actual
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