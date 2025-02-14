# test_crew.py
from crewai import Crew
from crew import LeadGenerationCrew  # Importa tu clase
from utils import load_environment_variables, load_profile_data, UserProfile
from pydantic import ValidationError

load_environment_variables()  # Carga las variables de entorno

profile_data = load_profile_data()
if profile_data is None:
    print("Error: No profile data found.")
    exit()

try:
    validated_user_profile = UserProfile(**profile_data)
except ValidationError as e:
        print(f"Profile validation error: {e}")
        exit()


# Datos de entrada de prueba
inputs = {
    'province': 'Buenos Aires',
    'industry': 'Empresas de Software/SaaS',
    'company_name': 'NEXTSYNAPSE',  # Prueba con y sin nombre de empresa
    'keywords': 'inteligencia artificial, machine learning',
    'company_size': 'Pequeña (1-50 empleados)',
    'revenue': '$1M - $10M',
    'location': 'Ciudad Autónoma de Buenos Aires',
    'technologies': 'Python, AWS',
    'needs': 'Necesitan mejorar su sistema de recomendación',
    'user_keywords': 'ciencia de datos, machine learning, IA, agentes de IA, GenAI' #, user_keywords
}
#Asegurarse que los datos del perfil del usuario se pasen como strings
user_profile_dict = validated_user_profile.model_dump()
if user_profile_dict.get("website"):
    user_profile_dict["website"] = str(user_profile_dict["website"])

inputs.update(user_profile_dict) # Importante agregar el perfil.

crew = LeadGenerationCrew()
results = crew.run(inputs=inputs)  # Pasa los inputs

if results:
    print("Crew execution successful:")
    print(results) #Ahora results es el reporte en string
else:
    print("Crew execution failed.")