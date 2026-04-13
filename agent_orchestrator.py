"""Agent orchestration using LangGraph + CrewAI facade"""
from typing import TypedDict, Annotated, Sequence
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from crewai import Agent, Task, Crew


class AgentState(TypedDict):
    """State schema for LangGraph agent workflow"""
    messages: Sequence[BaseMessage]
    current_task: str | None
    task_result: dict | None
    agent_type: str
    metadata: dict
    checkpoint_id: str | None


class OrchestrationConfig:
    """Configuration for agent orchestration"""
    
    def __init__(self, config_path: str = "config/orchestration.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> dict:
        import os
        config = {}
        if os.path.exists(self.config_path):
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        return config
    
    @property
    def persistence_config(self) -> dict:
        return self._config.get("orchestration", {}).get("persistence", {})
    
    @property
    def llm_config(self) -> dict:
        return self._config.get("llm", {})


class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, role: str, goal: str, config: OrchestrationConfig):
        self.name = name
        self.role = role
        self.goal = goal
        self.config = config
        self.state_history = []
    
    def execute(self, task: dict) -> dict:
        """Execute a task - to be implemented by subclasses"""
        raise NotImplementedError
    
    def save_checkpoint(self, state: AgentState) -> str:
        """Save state checkpoint to persistence layer"""
        checkpoint_id = f"checkpoint_{datetime.utcnow().timestamp()}"
        # Implementation for PostgreSQL/SQLite persistence
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> AgentState | None:
        """Load state from checkpoint"""
        # Implementation for PostgreSQL/SQLite persistence
        pass


class CrewAIFacade:
    """CrewAI facade for high-level agent configuration"""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self._crew = None
        self._agents = {}
    
    def create_agent(
        self,
        role: str,
        goal: str,
        tools: list[str] | None = None,
        backstory: str | None = None
    ) -> Agent:
        """Create a CrewAI agent with specified role and goal"""
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory or f"You are a {role} focused on {goal}",
            tools=tools or [],
            verbose=True
        )
        self._agents[role] = agent
        return agent
    
    def create_task(
        self,
        description: str,
        agent: Agent,
        expected_output: str | None = None
    ) -> Task:
        """Create a task for the crew"""
        return Task(
            description=description,
            agent=agent,
            expected_output=expected_output
        )
    
    def create_crew(
        self,
        agents: list[Agent],
        tasks: list[Task],
        process: str = "sequential"
    ) -> Crew:
        """Create a crew with agents and tasks"""
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=process,
            verbose=True
        )
        self._crew = crew
        return crew
    
    def execute(self, inputs: dict) -> dict:
        """Execute the crew workflow"""
        if not self._crew:
            raise RuntimeError("Crew not initialized")
        result = self._crew.kickoff(inputs=inputs)
        return {"result": result, "status": "completed"}


class LangGraphWorkflow:
    """LangGraph-based workflow for complex orchestration"""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph state machine"""
        
        def should_continue(state: AgentState) -> str:
            """Routing logic for workflow"""
            if state.get("task_result", {}).get("error"):
                return "retry"
            return "end"
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("code_scanner", self._code_scanner_node)
        workflow.add_node("knowledge_manager", self._knowledge_manager_node)
        workflow.add_node("compliance_auditor", self._compliance_auditor_node)
        
        workflow.set_entry_point("code_scanner")
        
        workflow.add_edge("code_scanner", "knowledge_manager")
        workflow.add_edge("knowledge_manager", "compliance_auditor")
        workflow.add_edge("compliance_auditor", END)
        
        self.graph = workflow.compile()
    
    async def _code_scanner_node(self, state: AgentState) -> AgentState:
        """Code scanning node"""
        task = state.get("current_task", {})
        result = {
            "scanner_results": [],
            "vulnerabilities": [],
            "pdp_violations": []
        }
        return {**state, "task_result": result}
    
    async def _knowledge_manager_node(self, state: AgentState) -> AgentState:
        """Knowledge base query node"""
        return {**state, "task_result": {}}
    
    async def _compliance_auditor_node(self, state: AgentState) -> AgentState:
        """Compliance audit node"""
        return {**state, "task_result": {}}
    
    async def execute(self, initial_state: AgentState) -> AgentState:
        """Execute the workflow"""
        return await self.graph.ainvoke(initial_state)


class AgentOrchestrator:
    """Main orchestrator combining LangGraph and CrewAI"""
    
    def __init__(self, config_path: str = "config/orchestration.yaml"):
        self.config = OrchestrationConfig(config_path)
        self.langgraph = LangGraphWorkflow(self.config)
        self.crewai_facade = CrewAIFacade(self.config)
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize agents from config"""
        agents_config = self.config._config.get("agents", {})
        
        for agent_name, agent_config in agents_config.items():
            self.crewai_facade.create_agent(
                role=agent_config["role"],
                goal=agent_config["goal"],
                tools=agent_config.get("tools", [])
            )
    
    async def execute_task(self, task: dict, use_crewai: bool = False) -> dict:
        """Execute a task using either LangGraph or CrewAI facade"""
        
        if use_crewai:
            return self.crewai_facade.execute(task)
        
        initial_state: AgentState = {
            "messages": [],
            "current_task": task.get("description"),
            "task_result": None,
            "agent_type": task.get("agent_type", "code_scanner"),
            "metadata": {"started_at": datetime.utcnow().isoformat()},
            "checkpoint_id": None
        }
        
        result = await self.langgraph.execute(initial_state)
        return result


if __name__ == "__main__":
    import asyncio
    
    orchestrator = AgentOrchestrator()
    
    task = {
        "description": "Scan repository for security vulnerabilities",
        "agent_type": "code_scanner"
    }
    
    result = asyncio.run(orchestrator.execute_task(task))
    print(json.dumps(result, default=str))