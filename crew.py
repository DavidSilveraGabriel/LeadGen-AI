from crewai import Crew, Process, Task, LLM  # Importa LLM
from crewai.project import CrewBase, agent, task, crew
from agents import (
    BusinessResearcherAgent,
    SalesCopywriterAgent,
    ReportingAnalystAgent,
    InformationExtractorAgent
)  # Importa tus agentes
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from agents import CheckLeadExistsTool, SearchLeadTool  # ¡No olvides las herramientas!
from utils import logger  # Importa el logger
from typing import List, Dict, Any
import os

# Configura el LLM para Gemini *aquí*, usando litellm.
# ¡MUY IMPORTANTE!  Asegúrate de que tu variable de entorno
# GEMINI_API_KEY esté configurada correctamente.
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash-exp",  # gemini/gemini-1.5-pro-latest O el modelo que quieras
    api_key=os.environ.get("GEMINI_API_KEY"), #Usa la api key desde las variables de entorno
    temperature=0.7, #Ajusta los parametros
    # max_tokens=4096,  #  Ajusta si es necesario
    # top_p=1.0,         #  Ajusta si es necesario
    # frequency_penalty=0.0, # Ajusta
    # presence_penalty=0.0,  # Ajusta
)


@CrewBase
class LeadGenerationCrew:
    """Crew para la generación de leads."""

    agents_config = "config/agents.yaml"  # Ruta a la configuración de agentes
    tasks_config = "config/tasks.yaml"  # Ruta a la configuración de tareas

    @agent
    def information_extractor_agent(self):
        return InformationExtractorAgent(llm=gemini_llm) #Pasa la instancia de LLM

    @agent
    def business_researcher(self):
        return BusinessResearcherAgent(tools=[SerperDevTool(), ScrapeWebsiteTool()], llm=gemini_llm) #Pasa la instancia

    @agent
    def sales_copywriter(self):
        return SalesCopywriterAgent(llm=gemini_llm) #Pasa la instancia

    @agent
    def reporting_analyst(self):
        return ReportingAnalystAgent(llm=gemini_llm) #Pasa la instancia

    @task
    def research_task(self): # Nombre del método: research_task
        return Task(config=self.tasks_config["research_business_task"])

    @task
    def email_task(self): # Nombre del método: email_task
        return Task(config=self.tasks_config["create_sales_email_task"])

    @task
    def report_task(self):# Nombre del método: report_task
        return Task(config=self.tasks_config["create_report_task"], output_file="output/report.md")

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # Automáticamente creado por @agent
            tasks=self.tasks,  # Automáticamente creado por @task
            process=Process.sequential,
            verbose=True,  # Aumenta la verbosidad para debugging
        )
