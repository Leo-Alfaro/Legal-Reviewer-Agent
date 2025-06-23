import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "key.env")
load_dotenv(dotenv_path)

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import fitz
from langchain_openai import ChatOpenAI

openai3_llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2
)

openai4_llm = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2
)

class MyReadTool(BaseModel):
   argument: str = Field(..., description="The path to the PDF file to be read.")

class FileReadTool(BaseTool):
    name: str = "FileReadTool"
    description: str = "A tool to read PDF files and extract text content. Use for any file path input."
    args_schema: type[BaseModel] = MyReadTool

    def _run(self, argument: str) -> str:
        try:
            doc = fitz.open(argument)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

@CrewBase
class MyAgentsCrew():
  agents: List[BaseAgent]
  tasks: List[Task]


  @agent
  def extractor(self) -> Agent:
    file_tool = FileReadTool()
    return Agent(
      config=self.agents_config['extractor'],
      verbose=True,
      tools=[file_tool],
      llm = openai3_llm,
    )
  
  @agent
  def contract_analyser(self) -> Agent:
    serper_tool = SerperDevTool()
    return Agent(
      config=self.agents_config['contract_analyser'],
      verbose=True,
      tools=[serper_tool],
      async_execution=True,
      llm = openai4_llm,
    )
  
  @agent
  def case_finder(self) -> Agent:
    serper_tool = SerperDevTool()
    return Agent(
      config=self.agents_config['case_finder'],
      verbose=True,
      tools=[serper_tool],
      async_execution=True,
      llm = openai4_llm,
    )
  
  @agent
  def Contract_Lawyer(self) -> Agent:
    return Agent(
      config=self.agents_config['Contract_Lawyer'],
      verbose=True,
      llm = openai4_llm,
    )

  @task
  def extract_info(self) -> Task:
    return Task(
      config=self.tasks_config['extract_info'],
    )
  
  @task
  def contract_analyser_info(self) -> Task:
    return Task(
      config=self.tasks_config['contract_analyser_info'],
    )

  @task
  def case_finder_info(self) -> Task:
    return Task(
      config=self.tasks_config['case_finder_info'],
    )
  
  @task
  def contract_lawyer_info(self) -> Task:
    return Task(
      config=self.tasks_config['contract_lawyer_info'],
    )

  @crew
  def crew(self) -> Crew:
    return Crew(
      agents=self.agents,
      tasks=self.tasks,
      process=Process.sequential,
      verbose=True,
    )