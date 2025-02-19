from crew import LeadGenerationCrew
from utils import load_environment_variables, load_profile_data, UserProfile
from pydantic import ValidationError

load_environment_variables()

profile_data = load_profile_data()
if profile_data is None:
    print("Error: No profile data found.")
    exit()
try:
    validated_user_profile = UserProfile(**profile_data)
    user_profile_dict = validated_user_profile.model_dump()
    # Convertir HttpUrl a str ANTES de pasar los inputs
    if user_profile_dict.get("website"):
        user_profile_dict["website"] = str(user_profile_dict["website"])
except ValidationError as e:
    print(f"Profile validation error: {e}")
    exit()

inputs = {
    'province': 'Buenos Aires',
    'industry': 'Empresas de Software/SaaS',
    'company_name': '',  # Prueba con y sin
    'keywords': 'inteligencia artificial, machine learning',
    'company_size': 'Pequeña (1-50 empleados)',
    'revenue': '$1M - $10M',
    'location': 'Ciudad Autónoma de Buenos Aires',
    'technologies': 'Python, AWS',
    'needs': 'Necesitan mejorar su sistema de recomendación',
    'user_keywords': 'ciencia de datos, machine learning, IA, agentes de IA',
    "summary": "Soy un **Científico de Datos e Ingeniero de Machine learning e Inteligencia Artificial** con más de cuatro años de experiencia en la creación de **soluciones de inteligencia artificial** innovadoras. Mi pasión radica en aplicar la tecnología para generar un **impacto positivo** y transformar la forma en que las organizaciones abordan sus desafíos.\nActualmente, desarrollo mi trabajo en **Argentina** y puedes encontrar más información sobre mis proyectos y experiencia profesional en:\n\n*   **Sitio web:** https://silveradavid.site/\n*   **LinkedIn:** https://www.linkedin.com/in/davidsilveragabriel/\n*   **GitHub:** https://github.com/DavidSilveraGabriel\n\nSi tienes alguna consulta o te gustaría conectar, puedes escribirme a ingenieria.d.s.g@hotmail.com\n\nMe especializo en:\n\n*   Implementación de modelos de lenguaje grandes (LLMs) para optimizar procesos organizacionales.\n*   Desarrollo de agentes personalizados para resolver desafíos tecnológicos.\n*   Visión por computadora.\n*   Análisis de datos",

}

#inputs.update(user_profile_dict)

crew = LeadGenerationCrew()  # Usamos la configuración por defecto
results = crew.run(inputs=inputs)

if results:
    print("Crew execution successful:")
    print(results)
else:
    print("Crew execution failed.")