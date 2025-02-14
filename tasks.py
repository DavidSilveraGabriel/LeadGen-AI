# tasks.py  
from crewai import Agent, Task
from typing import Dict, Any
from utils import logger

# Configuración básica de logging (si no la tienes ya)
#logging.basicConfig(
#    level=logging.DEBUG,
#    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
#) #Se comenta porque ahora se configura desde utils
#logger = logging.getLogger(__name__) # Se comenta porque ahora se importa desde utils

class AppTasks:
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.context = {} #Inicializamos context

    def research_business_task(self) -> Task:
      return Task(
          description="Investigación detallada de empresas.",
          expected_output="Datos enriquecidos sobre empresas, incluyendo contacto.",
          agent=self.agents['business_researcher'],
          context=[] # No pasar function_call.  CrewAI se encarga.
      )

    def create_sales_email_task(self) -> Task:
        return Task(
            description="Crea un correo de ventas personalizado.",
            expected_output="Correo electrónico en texto plano, listo para enviar.",
            agent=self.agents['sales_copywriter'],
            context=[self.agents['business_researcher']]
        )

    def create_report_task(self) -> Task:
        return Task(
            description="Consolida y guarda la información.",
            expected_output="Datos guardados y reporte final.",
            agent=self.agents['reporting_analyst'],
            context=[self.agents['sales_copywriter']]
        )