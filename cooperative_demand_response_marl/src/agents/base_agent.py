#!/usr/bin/env python3
"""
Classe base para agentes MARL no sistema de demand response cooperativo.

Este módulo implementa a classe BaseAgent que serve como interface comum
para todos os tipos de agentes MARL (independente, cooperativo, centralizado).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class BaseAgent(ABC):
    """
    Classe base abstrata para agentes MARL.

    Esta classe define a interface comum que todos os agentes devem implementar,
    incluindo métodos para seleção de ações, atualização de políticas e comunicação.

    Args:
        env: Ambiente de simulação (CityLearnVecEnv)
        agent_id: ID único do agente (0, 1, 2, ...)
        config: Configurações específicas do agente

    Attributes:
        env: Referência para o ambiente
        agent_id: ID único do agente
        config: Configurações do agente
        policy: Política de aprendizado (será definida pelas subclasses)
        training_history: Histórico de treinamento
        communication_buffer: Buffer para mensagens de comunicação
        episode_reward: Recompensa acumulada do episódio atual
        episode_length: Comprimento do episódio atual
    """

    def __init__(self, env, agent_id: int, config: Dict):
        """
        Inicializa o agente base.

        Args:
            env: Ambiente de simulação
            agent_id: ID único do agente
            config: Configurações do agente
        """
        self.env = env
        self.agent_id = agent_id
        self.config = config

        # Componentes do agente
        self.policy = None
        self.training_history = []
        self.communication_buffer = []

        # Estado do episódio
        self.episode_reward = 0.0
        self.episode_length = 0
        self.total_timesteps = 0

        # Configurações padrão
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)

        print(f"✅ BaseAgent {agent_id} inicializado")

    @abstractmethod
    def select_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Seleciona ação baseada na observação.

        Args:
            observation: Observação do ambiente
            **kwargs: Argumentos adicionais (ex: mensagens de comunicação)

        Returns:
            np.ndarray: Ação selecionada
        """
        pass

    @abstractmethod
    def update_policy(self, experience: Tuple, **kwargs) -> None:
        """
        Atualiza política baseada na experiência.

        Args:
            experience: Tupla (obs, action, reward, next_obs, done)
            **kwargs: Argumentos adicionais para atualização
        """
        pass

    def communicate(self, message: Dict) -> None:
        """
        Envia mensagem para outros agentes.

        Args:
            message: Dicionário com a mensagem
        """
        self.communication_buffer.append(message)

    def receive_messages(self) -> List[Dict]:
        """
        Recebe mensagens de outros agentes.

        Returns:
            List[Dict]: Lista de mensagens recebidas
        """
        messages = self.communication_buffer.copy()
        self.communication_buffer.clear()
        return messages

    def reset_episode(self) -> None:
        """Reseta estado do episódio."""
        self.episode_reward = 0.0
        self.episode_length = 0

    def update_episode(self, reward: float) -> None:
        """
        Atualiza estado do episódio.

        Args:
            reward: Recompensa recebida
        """
        self.episode_reward += reward
        self.episode_length += 1

    def get_info(self) -> Dict:
        """
        Retorna informações do agente.

        Returns:
            Dict: Informações do agente
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "total_timesteps": self.total_timesteps,
            "policy_type": type(self.policy).__name__ if self.policy else None,
            "training_steps": len(self.training_history)
        }

    def save_model(self, filepath: str) -> None:
        """
        Salva modelo do agente.

        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.policy and hasattr(self.policy, 'save'):
            self.policy.save(filepath)
            print(f"✅ Modelo do agente {self.agent_id} salvo em {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Carrega modelo do agente.

        Args:
            filepath: Caminho do modelo a carregar
        """
        if self.policy and hasattr(self.policy, 'load'):
            self.policy.load(filepath)
            print(f"✅ Modelo do agente {self.agent_id} carregado de {filepath}")

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Valida e ajusta ação se necessário.

        Args:
            action: Ação a validar

        Returns:
            np.ndarray: Ação validada
        """
        # Clipping para garantir que a ação está dentro dos limites
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        # Verificar se a ação é válida
        if not self.env.action_space.contains(action):
            warnings.warn(f"Agente {self.agent_id}: Ação fora do espaço válido: {action}")
            # Usar ação aleatória como fallback
            action = self.env.action_space.sample()

        return action

    def _log_training_step(self, experience: Tuple, loss: Optional[float] = None) -> None:
        """
        Registra passo de treinamento.

        Args:
            experience: Experiência de treinamento
            loss: Loss da atualização (se disponível)
        """
        obs, action, reward, next_obs, done = experience

        step_info = {
            "timestep": self.total_timesteps,
            "episode_length": self.episode_length,
            "episode_reward": self.episode_reward,
            "reward": reward,
            "loss": loss,
            "done": done
        }

        self.training_history.append(step_info)
        self.total_timesteps += 1

        # Manter apenas últimos 10000 passos para evitar uso excessivo de memória
        if len(self.training_history) > 10000:
            self.training_history.pop(0)

    def get_training_stats(self) -> Dict:
        """
        Retorna estatísticas de treinamento.

        Returns:
            Dict: Estatísticas de treinamento
        """
        if not self.training_history:
            return {"total_steps": 0, "mean_reward": 0.0, "mean_loss": 0.0}

        rewards = [step["reward"] for step in self.training_history]
        losses = [step["loss"] for step in self.training_history if step["loss"] is not None]

        return {
            "total_steps": self.total_timesteps,
            "training_steps": len(self.training_history),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_loss": np.mean(losses) if losses else 0.0,
            "final_episode_reward": self.episode_reward,
            "final_episode_length": self.episode_length
        }

    def __repr__(self) -> str:
        """Representação string do agente."""
        return (f"{self.__class__.__name__}(id={self.agent_id}, "
                f"policy={type(self.policy).__name__ if self.policy else None}, "
                f"steps={self.total_timesteps})")

    def __str__(self) -> str:
        """String descritiva do agente."""
        info = self.get_info()
        return (f"Agente {info['agent_id']} ({info['agent_type']})\n"
                f"  - Recompensa do episódio: {info['episode_reward']:.3f}\n"
                f"  - Comprimento do episódio: {info['episode_length']}\n"
                f"  - Passos totais: {info['total_timesteps']}\n"
                f"  - Política: {info['policy_type']}")


class AgentFactory:
    """
    Factory para criar diferentes tipos de agentes.

    Esta classe centraliza a criação de agentes e permite configuração
    flexível baseada em arquivos YAML ou dicionários de configuração.
    """

    @staticmethod
    def create_agent(agent_type: str, env, agent_id: int, config: Dict, **kwargs) -> BaseAgent:
        """
        Cria agente do tipo especificado.

        Args:
            agent_type: Tipo de agente ("independent", "cooperative", "centralized")
            env: Ambiente de simulação
            agent_id: ID do agente
            config: Configurações do agente
            **kwargs: Argumentos adicionais

        Returns:
            BaseAgent: Agente configurado
        """
        if agent_type == "independent":
            from .independent_agent import IndependentAgent
            return IndependentAgent(env, agent_id, config)
        elif agent_type == "cooperative":
            from .cooperative_agent import CooperativeAgent
            return CooperativeAgent(env, agent_id, config, **kwargs)
        elif agent_type == "centralized":
            from .centralized_agent import CentralizedAgent
            return CentralizedAgent(env, agent_id, config)
        else:
            raise ValueError(f"Tipo de agente inválido: {agent_type}")

    @staticmethod
    def create_multi_agent_system(env, config: Dict) -> List[BaseAgent]:
        """
        Cria sistema multi-agente completo.

        Args:
            env: Ambiente de simulação
            config: Configurações do sistema

        Returns:
            List[BaseAgent]: Lista de agentes criados
        """
        agent_type = config.get("agent_type", "independent")
        num_agents = env.num_buildings

        agents = []
        for i in range(num_agents):
            agent_config = config.copy()
            agent_config["agent_id"] = i

            # Configurações específicas por agente
            if "per_agent_config" in config and i in config["per_agent_config"]:
                agent_config.update(config["per_agent_config"][i])

            agent = AgentFactory.create_agent(agent_type, env, i, agent_config)
            agents.append(agent)

        print(f"✅ Sistema multi-agente criado: {len(agents)} agentes do tipo {agent_type}")
        return agents

    @staticmethod
    def load_config_from_yaml(config_path: str) -> Dict:
        """
        Carrega configuração de arquivo YAML.

        Args:
            config_path: Caminho do arquivo YAML

        Returns:
            Dict: Configuração carregada
        """
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def validate_config(config: Dict) -> bool:
        """
        Valida configuração do agente.

        Args:
            config: Configuração a validar

        Returns:
            bool: True se válida
        """
        required_fields = [
            "agent_type",
            "policy",
            "training"
        ]

        for field in required_fields:
            if field not in config:
                raise ValueError(f"Campo obrigatório ausente: {field}")

        # Validar tipo de agente
        valid_types = ["independent", "cooperative", "centralized"]
        if config["agent_type"] not in valid_types:
            raise ValueError(f"Tipo de agente inválido: {config['agent_type']}")

        # Validar algoritmo
        valid_algorithms = ["PPO", "SAC", "A2C", "DQN"]
        algorithm = config["training"].get("algorithm", "PPO")
        if algorithm not in valid_algorithms:
            raise ValueError(f"Algoritmo inválido: {algorithm}")

        return True


# Funções utilitárias para análise de agentes
def analyze_agent_performance(agents: List[BaseAgent], env, num_episodes: int = 10) -> Dict:
    """
    Analisa performance de múltiplos agentes.

    Args:
        agents: Lista de agentes a analisar
        env: Ambiente de simulação
        num_episodes: Número de episódios para análise

    Returns:
        Dict: Métricas de performance
    """
    results = {}

    for agent in agents:
        print(f"Analisando agente {agent.agent_id} ({agent.__class__.__name__})...")

        episode_rewards = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        # Calcular estatísticas
        rewards = np.array(episode_rewards)
        results[agent.agent_id] = {
            "agent_type": agent.__class__.__name__,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "total_episodes": num_episodes
        }

        print(f"   - Recompensa média: {results[agent.agent_id]['mean_reward']:.3f}")

    return results


def compare_agents(agents: List[BaseAgent], env, num_episodes: int = 10) -> Dict:
    """
    Compara performance entre diferentes agentes.

    Args:
        agents: Lista de agentes a comparar
        env: Ambiente de simulação
        num_episodes: Número de episódios para comparação

    Returns:
        Dict: Resultados da comparação
    """
    results = analyze_agent_performance(agents, env, num_episodes)

    # Calcular ranking
    agent_ranks = sorted(results.items(),
                        key=lambda x: x[1]["mean_reward"],
                        reverse=True)

    comparison = {
        "individual_results": results,
        "ranking": [(agent_id, info["mean_reward"]) for agent_id, info in agent_ranks],
        "best_agent": agent_ranks[0][0],
        "worst_agent": agent_ranks[-1][0],
        "performance_range": agent_ranks[0][1]["mean_reward"] - agent_ranks[-1][1]["mean_reward"]
    }

    print("\n🏆 Ranking dos Agentes:")
    for i, (agent_id, reward) in enumerate(comparison["ranking"], 1):
        agent_type = results[agent_id]["agent_type"]
        print(f"   {i}. Agente {agent_id} ({agent_type}): {reward:.3f}")

    return comparison