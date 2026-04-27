from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# Import your custom tools
from legal_team.tools.custom_tool import LlamaParseTool, PineconeSearchTool

@CrewBase
class LegalTeam():
    """Automated Contract Risk Analysis Crew for Freelance Tech Professionals"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # ==========================================
    # AGENTS DEFINITION
    # ==========================================

    @agent
    def extractor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['extractor_agent'], # type: ignore[index]
            verbose=True,
            tools=[LlamaParseTool()]
        )

    @agent
    def analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher_agent'], # type: ignore[index]
            verbose=True,
            tools=[PineconeSearchTool()]
        )

    @agent
    def critic_explainer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['critic_explainer_agent'], # type: ignore[index]
            verbose=True
        )

    # ==========================================
    # TASKS DEFINITION
    # ==========================================

    @task
    def extract_contract_data_task(self) -> Task:
        return Task(
            config=self.tasks_config['extract_contract_data_task'] # type: ignore[index]
        )

    @task
    def analyze_contract_risks_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_contract_risks_task'] # type: ignore[index]
        )

    @task
    def research_legal_precedents_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_legal_precedents_task'] # type: ignore[index]
        )

    @task
    def generate_final_report_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_final_report_task'], # type: ignore[index]
            # Note: We already defined output_file='output/final_contract_risk_report.md' in tasks.yaml
            # However, you can also pass it here explicitly as output_file='output/final_contract_risk_report.md'
        )

    # ==========================================
    # CREW ORCHESTRATION
    # ==========================================

    @crew
    def crew(self) -> Crew:
        """Creates the Contract Analysis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )