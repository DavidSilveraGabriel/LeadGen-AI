# Lead Generation Agent: A Comprehensive CrewAI Application Powered by Gemini

This project showcases a sophisticated lead generation system built using CrewAI and powered by Google's Gemini large language model (LLM).  It automates the process of identifying and qualifying potential leads, crafting personalized sales outreach, and consolidating the findings into actionable reports, all within a streamlined and efficient workflow. This document provides a detailed overview of the system's architecture, functionality, and technical implementation.

## System Overview

The lead generation system operates on a multi-agent architecture orchestrated by CrewAI. Each agent specializes in a specific task, working collaboratively to achieve the overall goal of lead generation. The process comprises four key stages:

1. **User Profile Extraction (`InformationExtractorAgent`):**  The system begins by processing user-provided input, which might include a description of their services, target industries, keywords, and desired geographic location (currently focused on Argentina). The `InformationExtractorAgent` meticulously extracts and structures this information, creating a comprehensive user profile that serves as the foundation for subsequent stages.  This agent ensures data consistency and accuracy throughout the process, reducing the risk of errors stemming from inconsistent or incomplete input.

2. **Lead Research and Qualification (`BusinessResearcherAgent`):** Leveraging a combination of powerful tools, including a web search tool (`SerperDevTool`) and a web scraping tool (`ScrapeWebsiteTool`), the `BusinessResearcherAgent` systematically identifies and qualifies potential leads. The research is targeted towards companies operating within the specified industry and region, prioritizing those exhibiting a strong potential need for the user's services. This qualification process goes beyond simple keyword matching; it incorporates contextual analysis to identify companies whose activities or expressed needs align with the user's offerings.  The research utilizes a dynamically generated prompt (see `build_research_prompt` in `utils.py`) to ensure precision and relevance.  The output consists of a structured dataset containing detailed information for each identified lead, including company name, website URL, a concise description, and contact information (email addresses and social media links).  Error handling and retry mechanisms are implemented to ensure robustness in the face of network issues or temporary data unavailability (see `retry_with_logging` in `utils.py`).

3. **Personalized Sales Email Generation (`SalesCopywriterAgent`):**  The `SalesCopywriterAgent` leverages the enriched lead data from the previous stage to craft highly personalized sales emails.  These emails go beyond generic templates, incorporating specific details about each lead's business, challenges, and potential needs.  The agent meticulously tailors the message to resonate with each recipient, enhancing the likelihood of engagement and positive response.  The emails maintain a professional yet approachable tone and always include a clear call to action, guiding the recipient towards the next step in the sales process.

4. **Data Consolidation, Validation, and Reporting (`ReportingAnalystAgent`):** Finally, the `ReportingAnalystAgent` consolidates all gathered information, ensuring data consistency and accuracy.  This stage involves rigorous validation using Pydantic models (not explicitly shown but implied in the README) before saving the validated lead data to a Supabase database (database interaction managed by `utils.py`).  A comprehensive report is generated, summarizing all key findings and providing a clear overview of the identified leads and the personalized outreach strategy.

## Technical Architecture

* **Core Framework:** CrewAI provides the workflow orchestration and agent management.
* **Large Language Model (LLM):** Google Gemini powers the natural language processing tasks within each agent, enabling personalized communication and intelligent data analysis.
* **Data Storage:** Supabase, a PostgreSQL-based database, securely stores the collected lead data.  Connection and interaction details are handled by functions within `utils.py`.
* **Programming Language:** Python, leveraging its extensive libraries for web scraping, data processing, and LLM integration.
* **Key Libraries:** crewai, crewai-tools, google-generativeai, streamlit, requests, psycopg2-binary, python-dotenv, beautifulsoup4, re, and other libraries listed in `requirements.txt`.


## Installation and Deployment

1. **Virtual Environment:** Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows
   ```
2. **Install Dependencies:** Install the project's dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables:** Create a `.env` file containing API keys for Gemini and Supabase, and any other necessary credentials.  **Do not commit this file to your Git repository.**  The `load_environment_variables()` function in `utils.py` handles loading these variables.
4. **Run the Application:**  Use Streamlit to launch the application's user interface:
   ```bash
   streamlit run app.py
   ```

## Future Enhancements

* **Scalability:**  Implement strategies to handle larger datasets and improve processing speed for high-volume lead generation.
* **Advanced Lead Scoring:** Integrate a lead scoring system to prioritize leads based on their potential value and likelihood of conversion.
* **Multi-Lingual Support:** Extend the system's capabilities to support multiple languages, expanding its reach to a wider range of markets.
* **Integration with CRM:** Seamlessly integrate with popular CRM platforms to automate lead management and streamline sales processes.
* **Real-time Monitoring and Reporting:** Develop a dashboard to visualize key metrics, providing real-time insights into lead generation performance.


## Contact

For inquiries or collaborations, please contact: ingenieria.d.s.g@hotmail.com