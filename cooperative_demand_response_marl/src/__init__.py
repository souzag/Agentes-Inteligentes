"""
Cooperative Demand Response with Multi-Agent Reinforcement Learning
Módulo principal do projeto
"""

# Exportar submódulos principais
from . import environment
from . import agents
from . import algorithms
from . import utils

# Funções de conveniência
def make_citylearn_vec_env(dataset_name: str = "citylearn_challenge_2022_phase_1"):
    """Cria ambiente CityLearn vetorizado."""
    from .environment import make_citylearn_vec_env
    return make_citylearn_vec_env(dataset_name)

def create_agent_system(env, config: dict):
    """Cria sistema de agentes baseado na configuração."""
    from .agents import AgentFactory
    return AgentFactory.create_multi_agent_system(env, config)

__all__ = [
    "environment",
    "agents",
    "algorithms",
    "utils",
    "make_citylearn_vec_env",
    "create_agent_system"
]