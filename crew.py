#crew.py
from crewai import Crew, Task, LLM, Process
from crewai.project import CrewBase, agent, task
from agents import (
    BusinessResearcherAgent,
    SalesCopywriterAgent,
    ReportingAnalystAgent,
)
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from utils import logger
import os
from pydantic import BaseModel, Field

# Configura el LLM para Gemini
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash-exp",  # O el modelo que quieras
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.6,
) # gemini-2.0-flash-lite-preview-02-05, gemini-2.0-flash-exp, gemini-1.5-pro-latest

@CrewBase
class LeadGenerationCrew:
    """Crew para la generación de leads."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def business_researcher(self):
        return BusinessResearcherAgent(tools=[SerperDevTool(), ScrapeWebsiteTool()], llm=gemini_llm)

    @agent
    def sales_copywriter(self):
        return SalesCopywriterAgent(llm=gemini_llm)

    @agent
    def reporting_analyst(self):
        return ReportingAnalystAgent(llm=gemini_llm)

    @task
    def research_business_task(self):  # AHORA el nombre coincide con el YAML
        return Task(config=self.tasks_config["research_business_task"], expected_output="Lista de empresas...")

    @task
    def create_sales_email_task(self):  # AHORA el nombre coincide con el YAML
        return Task(config=self.tasks_config["create_sales_email_task"], expected_output="Borrador de correo...")

    @task
    def create_report_task(self):  # AHORA el nombre coincide con el YAML
        return Task(config=self.tasks_config["create_report_task"], output_file="output/report.md", expected_output="Informe final...")

    def run(self, inputs):
        logger.info("Iniciando LeadGenerationCrew.run con inputs: %s", inputs)

        # Crea las *instancias* de las tareas y agentes
        research_task_instance = self.research_business_task() #Usamos los metodos
        email_task_instance = self.create_sales_email_task()
        report_task_instance = self.create_report_task()
        researcher_instance = self.business_researcher()
        copywriter_instance = self.sales_copywriter()
        analyst_instance = self.reporting_analyst()


        # Crea la instancia de Crew
        crew = Crew(
            agents=[researcher_instance, copywriter_instance, analyst_instance],
            tasks=[research_task_instance, email_task_instance, report_task_instance],
            process=Process.sequential,
            verbose=True
        )

        # Ejecuta la crew
        try:
            results = crew.kickoff(inputs=inputs)
            logger.info("Crew completada. Resultados: %s", results)
            return results
        except Exception as e:
            logger.exception("Error durante la ejecución de la crew:")
            return None