import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Callable, TypeVar
import json
from supabase import create_client, Client
import logging
from pydantic import BaseModel, EmailStr, HttpUrl, ValidationError
from datetime import datetime
import functools
import time
import re


# --- Configuración de Logging ---
def setup_logger(name):
    """Configura un logger con formato y handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"leadgen_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger(__name__)

def load_environment_variables() -> None:
    load_dotenv()

load_environment_variables()

# --- Inicialización de Supabase ---
try:
    supabase_url: str = os.environ["SUPABASE_URL"]
    supabase_key: str = os.environ["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except KeyError as e:
    logger.error(f"Variable de entorno no encontrada: {e}")
    raise KeyError(f"Error: Variable de entorno no encontrada: {e}.") from e
except Exception as e:
    logger.error(f"Error al conectar con Supabase: {e}", exc_info=True)
    raise Exception(f"Error al conectar con Supabase: {e}") from e

# --- Modelos Pydantic (para validación de datos) ---

class CompanyData(BaseModel):
    """Modelo para los datos de la empresa."""
    company_name: str
    industry: str
    province: str
    website: Optional[HttpUrl] = None  # URL, validada por Pydantic
    email: Optional[EmailStr] = None # Email, validado
    instagram: Optional[str] = None
    facebook: Optional[str] = None
    about: Optional[str] = None
    source: str
    fecha_consulta: str


class EmailData(BaseModel):
    """Modelo para los datos del correo electrónico."""
    email_subject: str
    email_body: str
    keywords: List[str]
    generated_at: str

class UserProfile(BaseModel):
    """Modelo de Pydantic para el perfil del usuario."""
    name: Optional[str] = None
    age: Optional[int] = None
    company_name: Optional[str] = None
    role: Optional[str] = None
    website: Optional[HttpUrl] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    keywords: List[str] = []
    summary: Optional[str] = None
    interests: List[str] = []
    parsing_success: bool = False #Campo que indica si el llm logro parsear bien el texto, sino es falso

# --- Decorador para Reintentos y Manejo de Errores ---
RT = TypeVar('RT')  # Tipo de retorno genérico

def retry_with_logging(func: Callable[..., RT], max_retries: int = 3, retry_delay: int = 10,
                       allowed_exceptions: tuple = ()) -> Callable[..., RT]:
    """Decorador para reintentar una función y registrar errores."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> RT:
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except allowed_exceptions as e:
                logger.warning(
                    f"Error en {func.__name__} (intento {attempt + 1}/{max_retries}): {e}. "
                    f"Reintentando en {retry_delay} segundos..."
                )
                time.sleep(retry_delay)
                if attempt == max_retries - 1:
                    logger.error(f"Error en {func.__name__} después de {max_retries} intentos: {e}")
                    raise  # Relanza la excepción después del último intento
            except Exception as e:
                logger.exception(f"Error inesperado en {func.__name__}: {e}")  # Usa logger.exception
                raise
    return wrapper

# --- Funciones de Acceso a Datos (Supabase) (con reintentos) ---
def save_lead(lead_data: Dict[str, Any]) -> None:
    """Guarda un lead, ahora usando Pydantic para validación."""
    try:
        validated_data = CompanyData(**lead_data).model_dump()
        data, count = supabase.table("leads").insert(validated_data).execute()
        logger.info(f"Lead guardado en Supabase: {data}")
    except ValidationError as e:
        logger.error(f"Error de validación al guardar el lead: {e}")
        raise
    except Exception as e:
        logger.error(f"Error al guardar el lead: {e}", exc_info=True)
        raise

def _decorated_save_lead(*args, **kwargs):  # Función auxiliar PRIMERO
     return save_lead(*args, **kwargs)

save_lead = retry_with_logging(_decorated_save_lead, allowed_exceptions=(Exception,))  # Decorador y reasignación

def search_lead(company_name: str, province: str) -> Optional[Dict[str, Any]]:
    try:
        data, count = supabase.table("leads").select("*").eq("company_name", company_name).eq("province", province).execute()
        if data and len(data[1]) > 0:
            return data[1][0]
        return None
    except Exception as e:
        logger.error(f"Error al buscar el lead: {e}", exc_info=True)
        raise

def _decorated_search_lead(*args, **kwargs): #Auxiliar
     return search_lead(*args, **kwargs)
search_lead = retry_with_logging(_decorated_search_lead, allowed_exceptions=(Exception,)) # Decorador y reasignación

def check_lead_exists(company_name: str, province: str) -> bool:
    try:
        data, count = supabase.table("leads").select("id").eq("company_name", company_name).eq("province", province).execute()
        return len(data[1]) > 0
    except Exception as e:
        logger.error(f"Error al verificar existencia: {e}", exc_info=True)
        raise

def _decorated_check_lead_exists(*args, **kwargs): #Auxiliar
     return check_lead_exists(*args, **kwargs)

check_lead_exists = retry_with_logging(_decorated_check_lead_exists, allowed_exceptions=(Exception,)) #Decorador

# --- Funciones de Archivo (Perfil) ---

def save_profile_data(data: Dict[str, Any], filename: str = "profile_data.json") -> None:
    """Guarda los datos del perfil en un archivo JSON."""
    logger.debug(f"Guardando datos del perfil en: {filename}")
    filepath = os.path.join("outputs", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convierte HttpUrl a str ANTES de guardar
    data_to_save = data.copy()  # Crea una copia para no modificar el original
    if "website" in data_to_save and data_to_save["website"] is not None:
        data_to_save["website"] = str(data_to_save["website"])

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)  # Usa la copia modificada
    logger.info(f"Datos del perfil guardados exitosamente en {filepath}")

def load_profile_data(filename: str = "profile_data.json") -> Optional[Dict[str, Any]]:
    logger.debug(f"Cargando datos del perfil desde: {filename}")
    filepath = os.path.join("outputs", filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Validación con Pydantic después de cargar
                validated_data = UserProfile(**data)  # Valida
                logger.info(f"Datos del perfil cargados y validados exitosamente desde {filepath}")
                return validated_data.model_dump()  # Devuelve como diccionario
            except ValidationError as e:
                logger.error(f"Error de validación al cargar el perfil: {e}")
                return None  # O considera lanzar la excepción
            except json.JSONDecodeError:
                logger.error(f"Error al decodificar JSON desde {filepath}")
                return None # O lanza el error

    else:
        logger.warning(f"Archivo de perfil no encontrado: {filepath}")
        return None


# --- Funciones de Prompting (NUEVAS) ---

# Usando un diccionario de fragmentos
def build_research_prompt(search_data: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
    """Construye el prompt para la búsqueda, combinando datos de búsqueda y perfil."""
    logger.debug("INICIO de build_research_prompt")

    prompt_fragments = {
        "base": f"Encuentra empresas en el rubro de {search_data.get('industry', 'empresas')} en {search_data.get('province', 'Argentina')}, Argentina.",
        "keywords": f"Estas empresas deberían estar relacionadas con: {', '.join(user_profile.get('keywords', []))}.",
        "summary": f"Considera la siguiente descripción de mis servicios: '{user_profile.get('summary', '')}'.",
        "instructions": (
            "Dame información detallada sobre cada empresa, incluyendo: nombre, sitio web (URL válida), "
            "una breve descripción, y cualquier información de contacto que encuentres (correo electrónico válido, redes sociales)."
             "Prioriza empresas que mencionen explicitamente la necesidad de servicios relacionados con las palabras clave proporcionadas"
        ),
    }

    prompt = " ".join(
        [
            prompt_fragments["base"],
            prompt_fragments["keywords"] if user_profile.get('keywords') else "",
            prompt_fragments["summary"] if user_profile.get('summary') else "",
            prompt_fragments["instructions"],
        ]
    )
    logger.debug(f"PROMPT-->{prompt}")
    logger.debug("FIN de build_research_prompt")

    return prompt


def build_email_prompt(company_data: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
    """Construye el prompt para el correo, MUCHO más específico."""
    user_name = user_profile.get('name', 'Representante de Ventas')
    user_company = user_profile.get('company_name', 'Nuestra Empresa')
    user_role = user_profile.get('role', 'Representante de Ventas')
    user_website = user_profile.get('website', 'No Disponible')
    user_keywords = ", ".join(user_profile.get('keywords', []))
    user_summary = user_profile.get('summary', '')

    company_name = company_data.get('company_name', 'la empresa')
    company_industry = company_data.get('industry', 'el sector')
    company_province = company_data.get('province', 'la región')
    company_about = company_data.get('about', 'la información disponible')

    prompt = (
        f"Escribe un correo electrónico de ventas altamente personalizado para {company_name}, "
        f"una empresa en el rubro de {company_industry} ubicada en {company_province}.\n\n"
        f"Información sobre {company_name}:\n{company_about}\n\n"
        f"Debes presentarte como {user_name} de {user_company}, {user_role}.\n"
        f"Tu sitio web es {user_website}.\n"
        f"Tus áreas de especialización son: {user_keywords}.\n"
        f"Breve descripción de tu experiencia: {user_summary}\n\n"
        f"El objetivo del correo es iniciar una conversación y programar una breve reunión "
        f"para discutir cómo tus habilidades/servicios pueden *específicamente* ayudar a {company_name} "
        f"a resolver sus desafíos o aprovechar oportunidades en su rubro.\n\n"
        f"Debes *conectar* la información sobre {company_name} con tus habilidades/servicios. "
        f"Por ejemplo, si la descripción de la empresa menciona un enfoque en 'SEO', y tú tienes "
        f"experiencia en 'generación de contenido con IA', debes mencionar cómo puedes ayudarles "
        f"a mejorar su SEO con contenido de alta calidad.\n\n"
        f"El tono debe ser profesional pero amigable, y el correo debe ser conciso (no más de 200 palabras).\n"
        f"Incluye una llamada a la acción (CTA) clara, proponiendo un tema específico para la reunión."
        f"El correo debe estar en formato de texto plano, listo para ser enviado. Evita saludos genéricos, sé directo"
    )
    return prompt

# --- Función para parsear la respuesta de Serper (implementar) ---

def _parse_serper_text_results(serper_results: str) -> dict:
    """
    Esta función debe parsear la respuesta en texto plano de SerperDevTool
    y convertirla a un diccionario. La implementación exacta dependerá del
    formato de la respuesta de Serper.
    """
    try:
        # Aquí va la lógica para extraer los datos relevantes.
        # Esto es un placeholder, necesitas la estructura real.
        # Podría usar expresiones regulares, o buscar patrones específicos.
        data = {}  # Inicializa un diccionario vacío

        # Ejemplo MUY básico (asumiendo que la respuesta es JSON-like):
        cleaned_results = serper_results.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_results)

        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON de Serper: {e}. Respuesta:\n{serper_results}")
        return {"error": "Respuesta de Serper no es JSON válido."}
    except Exception as e:
        logger.error(f"Error al parsear resultados de Serper: {e}")
        return {"error": "Error desconocido al parsear resultados de Serper."}