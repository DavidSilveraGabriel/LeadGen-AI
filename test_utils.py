import unittest
from unittest.mock import patch, MagicMock
from utils import _parse_serper_with_llm, logger  # Importa la función
from crewai import LLM  # Importa LLM si lo estás usando


class TestParseSerperWithLLM(unittest.TestCase):

    @patch('crewai.LLM.call')  # Usar crewai.LLM.call
    def test_valid_json_output(self, mock_llm_call):
        """Test successful parsing with valid JSON output from LLM."""
        # Simulate a successful response from the LLM
        mock_llm_call.return_value = """
        {
        "organic": [
            {"title": "Title 1", "link": "http://link1.com", "snippet": "Snippet 1", "source": "serper"},
            {"title": "Title 2", "link": "http://link2.com", "snippet": "Snippet 2", "source": "serper"}
        ],
            "insight": "Main trend: AI is cool."
        }
        """
        serper_results = "Some raw Serper output (doesn't matter in this case)"
        expected = {
            "organic": [
                {"title": "Title 1", "link": "http://link1.com", "snippet": "Snippet 1", "source": "serper"},
                {"title": "Title 2", "link": "http://link2.com", "snippet": "Snippet 2", "source": "serper"}
            ],
            "insight": "Main trend: AI is cool."
        }
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")  # Puedes usar un modelo falso aquí para el test
        result = _parse_serper_with_llm(agent, serper_results)
        self.assertEqual(result, expected)
        mock_llm_call.assert_called_once() #Verifica que se llamo al llm


    @patch('crewai.LLM.call')
    def test_invalid_json_output(self, mock_llm_call):
        """Test handling invalid JSON output from LLM."""
        mock_llm_call.return_value = "This is not valid JSON"
        serper_results = "Some raw Serper output"
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")  # Puedes usar un modelo falso aquí para el test
        result = _parse_serper_with_llm(agent, serper_results)
        self.assertIn("error", result)
        self.assertIn("not valid JSON", result["error"])
        mock_llm_call.assert_called_once() #Verifica que se llamo al llm

    @patch('crewai.LLM.call')
    def test_missing_organic_key(self, mock_llm_call):
        """Test LLM output missing the 'organic' key."""
        mock_llm_call.return_value = '{"other_key": []}'
        serper_results = "Some raw Serper output"
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")  # Puedes usar un modelo falso aquí para el test
        result = _parse_serper_with_llm(agent, serper_results)
        self.assertIn("error", result)
        self.assertIn("did not contain an 'organic' key", result["error"])
        mock_llm_call.assert_called_once() #Verifica que se llamo al llm

    @patch('crewai.LLM.call')
    def test_organic_not_list(self, mock_llm_call):
        """Test 'organic' key not being a list."""
        mock_llm_call.return_value = '{"organic": "not a list"}'
        serper_results = "Some raw Serper output"
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")
        result = _parse_serper_with_llm(agent,serper_results)
        self.assertIn("error", result)
        self.assertIn("is not a list", result["error"])
        mock_llm_call.assert_called_once() #Verifica que se llamo al llm

    @patch('crewai.LLM.call')
    @patch('utils.logger.exception')
    def test_unexpected_error(self, mock_logger, mock_llm_call):
        """Test handling unexpected errors during parsing."""
        mock_llm_call.side_effect = Exception("Unexpected error!") #Simulamos un error inesperado
        serper_results = "Some raw Serper output"
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")
        result = _parse_serper_with_llm(agent, serper_results)
        self.assertIn("error", result)
        self.assertIn("Unexpected error", result["error"])
        mock_logger.assert_called_once() #Verifica que se llamo al logger.exception
        mock_llm_call.assert_called_once() #Verifica que se llamo al llm

    @patch('crewai.LLM.call')
    def test_empty_organic_list(self, mock_llm_call):
        """Test handling an empty organic list from the LLM."""
        mock_llm_call.return_value = '{"organic": []}'
        serper_results = "Some raw Serper output"
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")
        result = _parse_serper_with_llm(agent, serper_results)
        expected = {"organic": [], "insight": None, "error": "LLM output did not contain an 'organic' key with a list value."} #Se espera este resultado

        self.assertEqual(result, expected) #Se comprueba
        mock_llm_call.assert_called_once() #Verifica que se llamo al llm

    @patch('crewai.LLM.call')
    def test_no_organic_key_but_error(self, mock_llm_call):
        """Test case where LLM returns an error message."""
        mock_llm_call.return_value = '{"error": "Could not retrieve results"}'
        serper_results = "Some raw Serper output"
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")
        result = _parse_serper_with_llm(agent, serper_results)
        expected = {"error": "Could not retrieve results"}
        self.assertEqual(result, expected)
        mock_llm_call.assert_called_once() #Verifica que se llamo al llm
    
    @patch('crewai.LLM.call')
    def test_valid_json_output_no_insight(self, mock_llm_call):
        """Test successful parsing with valid JSON output, but no insight provided"""
        mock_llm_call.return_value = """
        {
        "organic": [
            {"title": "Title 1", "link": "http://link1.com", "snippet": "Snippet 1", "source": "serper"},
            {"title": "Title 2", "link": "http://link2.com", "snippet": "Snippet 2", "source": "serper"}
        ]
        }
        """
        serper_results = "Salida de Serper (no importa)"
        #Simulamos que el agente tiene un llm.
        agent = MagicMock()
        agent.llm = LLM(model="gemini/gemini-2.0-flash-exp")
        result = _parse_serper_with_llm(agent,serper_results)
        expected =  {
        "organic": [
            {"title": "Title 1", "link": "http://link1.com", "snippet": "Snippet 1", "source": "serper"},
            {"title": "Title 2", "link": "http://link2.com", "snippet": "Snippet 2", "source": "serper"}
        ],
            "insight": None, #Se espera que insight sea None
            "error": "LLM output did not contain an 'organic' key with a list value."
        }
        self.assertEqual(result, {"organic": [], "insight": None, "error": "LLM output did not contain an 'organic' key with a list value."})


if __name__ == '__main__':
    unittest.main()