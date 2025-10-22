"""
Agentes MARL para demand response cooperativo.

Este módulo fornece uma implementação completa de agentes MARL (Multi-Agent
Reinforcement Learning) para o sistema de demand response cooperativo.

Classes principais:
- BaseAgent: Classe base abstrata para todos os agentes
- IndependentAgent: Agente que aprende independentemente
- CooperativeAgent: Agente que coopera com outros agentes
- CentralizedAgent: Agente centralizado que controla todos os prédios
- RandomAgent: Agente baseline que seleciona ações aleatórias
- RuleBasedAgent: Agente baseline baseado em regras

Políticas customizadas:
- MultiAgentPolicy: Política para múltiplos agentes
- CooperativePolicy: Política otimizada para cooperação
- CentralizedPolicy: Política para controle centralizado
- AttentionPolicy: Política baseada em mecanismos de atenção

Factories:
- AgentFactory: Factory principal para criação de agentes
- IndependentAgentFactory: Factory para agentes independentes
- CooperativeAgentFactory: Factory para agentes cooperativos
- CentralizedAgentFactory: Factory para agentes centralizados
- RandomAgentFactory: Factory para agentes aleatórios
- RuleBasedAgentFactory: Factory para agentes baseados em regras

Exemplo de uso:
    >>> from src.agents import AgentFactory
    >>> from src.environment import make_citylearn_vec_env
    >>>
    >>> env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")
    >>> agents = AgentFactory.create_multi_agent_system(env, {"agent_type": "cooperative"})
    >>>
    >>> # Treinamento
    >>> for agent in agents:
    ...     agent.train(total_timesteps=100000)
"""

# Classes de agentes
from .base_agent import BaseAgent, AgentFactory
from .independent_agent import IndependentAgent, IndependentAgentFactory
from .cooperative_agent import CooperativeAgent, CooperativeAgentFactory
from .centralized_agent import CentralizedAgent, CentralizedAgentFactory

# Agentes baseline
from .agent_factory import RandomAgent, RuleBasedAgent, RandomAgentFactory, RuleBasedAgentFactory

# Políticas customizadas
from .policies import (
    MultiAgentPolicy,
    CooperativePolicy,
    CentralizedPolicy,
    AttentionPolicy,
    MultiAgentFeaturesExtractor,
    register_custom_policies,
    create_policy_from_config,
    get_policy_info
)

# Factory principal
from .agent_factory import (
    MultiAgentFactory,
    create_agent_from_config,
    create_multi_agent_from_config,
    get_available_agent_types,
    get_default_config,
    validate_agent_config
)

__all__ = [
    # Classes de agentes
    "BaseAgent",
    "IndependentAgent",
    "CooperativeAgent",
    "CentralizedAgent",

    # Agentes baseline
    "RandomAgent",
    "RuleBasedAgent",

    # Factories
    "AgentFactory",
    "IndependentAgentFactory",
    "CooperativeAgentFactory",
    "CentralizedAgentFactory",
    "RandomAgentFactory",
    "RuleBasedAgentFactory",
    "MultiAgentFactory",

    # Políticas
    "MultiAgentPolicy",
    "CooperativePolicy",
    "CentralizedPolicy",
    "AttentionPolicy",
    "MultiAgentFeaturesExtractor",

    # Funções utilitárias
    "register_custom_policies",
    "create_policy_from_config",
    "get_policy_info",
    "create_agent_from_config",
    "create_multi_agent_from_config",
    "get_available_agent_types",
    "get_default_config",
    "validate_agent_config"
]