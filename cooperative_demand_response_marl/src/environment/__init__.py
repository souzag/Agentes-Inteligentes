"""
Ambiente de simulação para resposta cooperativa à demanda usando MARL.

Este módulo fornece uma implementação completa do ambiente vetorizado CityLearn
otimizado para treinamento de algoritmos multi-agente cooperativos com Stable Baselines3.

Classes principais:
- CityLearnVecEnv: Ambiente vetorizado principal
- LocalReward: Recompensa baseada em estado individual
- GlobalReward: Recompensa baseada em estado global
- CooperativeReward: Recompensa cooperativa (recomendada)
- AdaptiveReward: Recompensa adaptativa

Protocolos de comunicação:
- FullCommunication: Todos se comunicam com todos
- NeighborhoodCommunication: Apenas vizinhos
- CentralizedCommunication: Coordenação centralizada
- HierarchicalCommunication: Hierarquia de comunicação

Exemplo de uso:
    >>> from src.environment import make_citylearn_vec_env
    >>> env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, done, info = env.step(action)
"""

from .citylearn_vec_env import CityLearnVecEnv, make_citylearn_vec_env
from .rewards import (
    RewardFunction,
    LocalReward,
    GlobalReward,
    CooperativeReward,
    AdaptiveReward,
    create_reward_function,
    analyze_reward_components
)
from .communication import (
    CommunicationProtocol,
    CommunicationType,
    Message,
    FullCommunication,
    NeighborhoodCommunication,
    CentralizedCommunication,
    HierarchicalCommunication,
    create_communication_protocol,
    analyze_communication_network,
    optimize_communication_topology
)

__all__ = [
    # Ambiente principal
    "CityLearnVecEnv",
    "make_citylearn_vec_env",

    # Funções de recompensa
    "RewardFunction",
    "LocalReward",
    "GlobalReward",
    "CooperativeReward",
    "AdaptiveReward",
    "create_reward_function",
    "analyze_reward_components",

    # Comunicação
    "CommunicationProtocol",
    "CommunicationType",
    "Message",
    "FullCommunication",
    "NeighborhoodCommunication",
    "CentralizedCommunication",
    "HierarchicalCommunication",
    "create_communication_protocol",
    "analyze_communication_network",
    "optimize_communication_topology"
]