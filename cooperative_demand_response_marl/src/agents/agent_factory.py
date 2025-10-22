#!/usr/bin/env python3
"""
Factory functions para criação de agentes MARL.

Este módulo fornece funções factory para criar diferentes tipos de agentes
de forma padronizada e configurável, facilitando a experimentação e
comparação entre diferentes abordagens.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentFactory
from .independent_agent import IndependentAgent, IndependentAgentFactory
from .cooperative_agent import CooperativeAgent, CooperativeAgentFactory
from .centralized_agent import CentralizedAgent, CentralizedAgentFactory


class MultiAgentFactory:
    """
    Factory principal para criação de sistemas multi-agente.

    Esta classe centraliza a criação de diferentes tipos de agentes
    e sistemas multi-agente, permitindo configuração flexível e
    experimentação fácil.
    """

    @staticmethod
    def create_agent_system(env, config: Dict) -> Union[BaseAgent, List[BaseAgent]]:
        """
        Cria sistema de agentes baseado na configuração.

        Args:
            env: Ambiente de simulação
            config: Configurações do sistema

        Returns:
            Agente ou lista de agentes
        """
        agent_type = config.get("agent_type", "independent")

        if agent_type == "centralized":
            # Sistema centralizado: um agente controla todos
            return CentralizedAgentFactory.create_centralized_system(env, config)
        else:
            # Sistema multi-agente: múltiplos agentes
            return AgentFactory.create_multi_agent_system(env, config)

    @staticmethod
    def create_comparison_system(env, agent_types: List[str],
                               configs: Optional[Dict] = None) -> Dict[str, List[BaseAgent]]:
        """
        Cria sistemas de diferentes tipos para comparação.

        Args:
            env: Ambiente de simulação
            agent_types: Lista de tipos de agentes a criar
            configs: Configurações específicas por tipo

        Returns:
            Dict: Dicionário com sistemas de agentes por tipo
        """
        systems = {}

        for agent_type in agent_types:
            config = configs.get(agent_type, {}) if configs else {}
            config["agent_type"] = agent_type

            system = MultiAgentFactory.create_agent_system(env, config)
            systems[agent_type] = system

        print(f"✅ Sistemas de comparação criados: {list(systems.keys())}")
        return systems

    @staticmethod
    def create_baseline_agents(env) -> Dict[str, List[BaseAgent]]:
        """
        Cria agentes baseline para comparação.

        Args:
            env: Ambiente de simulação

        Returns:
            Dict: Dicionário com agentes baseline
        """
        baselines = {}

        # Agente aleatório
        baselines["random"] = RandomAgentFactory.create_multi_agent_system(env)

        # Agente baseado em regras
        baselines["rule_based"] = RuleBasedAgentFactory.create_multi_agent_system(env)

        # Agente independente PPO
        ppo_config = {
            "agent_type": "independent",
            "training": {"algorithm": "PPO", "total_timesteps": 100000}
        }
        baselines["independent_ppo"] = AgentFactory.create_multi_agent_system(env, ppo_config)

        print(f"✅ Agentes baseline criados: {list(baselines.keys())}")
        return baselines


class RandomAgent(BaseAgent):
    """
    Agente que seleciona ações aleatórias (baseline).

    Este agente serve como baseline simples para comparar com
    abordagens de aprendizado por reforço.
    """

    def __init__(self, env, agent_id: int, config: Dict):
        """Inicializa agente aleatório."""
        super().__init__(env, agent_id, config)
        print(f"✅ RandomAgent {agent_id} inicializado")

    def select_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Seleciona ação aleatória."""
        return self.env.action_space.sample()

    def update_policy(self, experience: Tuple, **kwargs) -> None:
        """Não faz nada (agente aleatório não aprende)."""
        self._log_training_step(experience, loss=0.0)

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Avalia performance do agente aleatório.

        Args:
            num_episodes: Número de episódios para avaliação

        Returns:
            Dict: Métricas de performance
        """
        print(f"📊 Avaliando RandomAgent {self.agent_id} por {num_episodes} episódios...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.select_action(obs)
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                # Adicionar logs para debugging
                if episode_length % 1000 == 0:
                    print(f"   - Episódio {episode+1}: Passo {episode_length}, Done: {done}, Recompensa: {episode_reward:.3f}")

                # Limitar comprimento do episódio com limite adicional
                if episode_length > 10000:
                    print(f"   - Episódio {episode+1}: Limite de passos atingido, interrompendo.")
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Calcular estatísticas
        rewards = np.array(episode_rewards)
        lengths = np.array(episode_lengths)

        results = {
            "agent_id": self.agent_id,
            "agent_type": "RandomAgent",
            "num_episodes": num_episodes,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "total_timesteps": self.total_timesteps
        }

        print(f"   - Recompensa média: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"   - Comprimento médio: {results['mean_length']:.1f} ± {results['std_length']:.1f}")

        return results


class RuleBasedAgent(BaseAgent):
    """
    Agente baseado em regras fixas (baseline).

    Este agente implementa regras simples para controle de HVAC
    baseado em thresholds de temperatura e outras heurísticas.
    """

    def __init__(self, env, agent_id: int, config: Dict):
        """
        Inicializa agente baseado em regras.

        Args:
            env: Ambiente de simulação
            agent_id: ID do agente
            config: Configurações
        """
        super().__init__(env, agent_id, config)

        # Configurações das regras
        self.temp_thresholds = config.get("temp_thresholds", {
            "cooling_start": 24.0,
            "cooling_stop": 22.0,
            "heating_start": 20.0,
            "heating_stop": 22.0
        })

        self.hvac_action = 0.0  # Estado atual do HVAC
        self.last_temp = None

        print(f"✅ RuleBasedAgent {agent_id} inicializado")

    def select_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Seleciona ação baseada em regras.

        Args:
            observation: Observação do ambiente
            **kwargs: Argumentos adicionais

        Returns:
            np.ndarray: Ação baseada em regras
        """
        # Extrair temperatura (assumindo que está nas primeiras features)
        if len(observation) >= 4:
            current_temp = observation[3]  # Temperatura interna

            # Regra de controle de temperatura
            if current_temp > self.temp_thresholds["cooling_start"]:
                # Iniciar resfriamento
                self.hvac_action = -0.5  # Resfriamento moderado
            elif current_temp < self.temp_thresholds["heating_start"]:
                # Iniciar aquecimento
                self.hvac_action = 0.5   # Aquecimento moderado
            elif (self.temp_thresholds["cooling_stop"] <= current_temp <= self.temp_thresholds["heating_stop"]):
                # Manter temperatura
                self.hvac_action = 0.0

            self.last_temp = current_temp
        else:
            # Fallback para ação aleatória
            self.hvac_action = np.random.uniform(-0.3, 0.3)

        # Adicionar pequeno ruído para exploração
        noise = np.random.normal(0, 0.1)
        action = np.array([self.hvac_action + noise])

        return self._validate_action(action)

    def update_policy(self, experience: Tuple, **kwargs) -> None:
        """Não faz nada (agente baseado em regras não aprende)."""
        self._log_training_step(experience, loss=0.0)

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Avalia performance do agente baseado em regras.

        Args:
            num_episodes: Número de episódios para avaliação

        Returns:
            Dict: Métricas de performance
        """
        print(f"📊 Avaliando RuleBasedAgent {self.agent_id} por {num_episodes} episódios...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action = self.select_action(obs)
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                # Adicionar logs para debugging
                if episode_length % 1000 == 0:
                    print(f"   - Episódio {episode+1}: Passo {episode_length}, Done: {done}, Recompensa: {np.mean(rewards):.3f}")

                # Limitar comprimento do episódio com limite adicional
                if episode_length > 10000:
                    print(f"   - Episódio {episode+1}: Limite de passos atingido, interrompendo.")
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Calcular estatísticas
        rewards = np.array(episode_rewards)
        lengths = np.array(episode_lengths)

        results = {
            "agent_id": self.agent_id,
            "agent_type": "RuleBasedAgent",
            "num_episodes": num_episodes,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "total_timesteps": self.total_timesteps
        }

        print(f"   - Recompensa média: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"   - Comprimento médio: {results['mean_length']:.1f} ± {results['std_length']:.1f}")

        return results


class RandomAgentFactory:
    """Factory para agentes aleatórios."""

    @staticmethod
    def create_agent(env, agent_id: int, config: Optional[Dict] = None) -> RandomAgent:
        """Cria agente aleatório."""
        default_config = {"agent_type": "random"}
        if config:
            default_config.update(config)
        return RandomAgent(env, agent_id, default_config)

    @staticmethod
    def create_multi_agent_system(env, num_agents: Optional[int] = None,
                                config: Optional[Dict] = None) -> List[RandomAgent]:
        """Cria sistema de agentes aleatórios."""
        if num_agents is None:
            num_agents = env.num_buildings

        agents = []
        for i in range(num_agents):
            agent_config = config.copy() if config else {}
            agent_config["agent_id"] = i
            agent = RandomAgentFactory.create_agent(env, i, agent_config)
            agents.append(agent)

        return agents


class RuleBasedAgentFactory:
    """Factory para agentes baseados em regras."""

    @staticmethod
    def create_agent(env, agent_id: int, config: Optional[Dict] = None) -> RuleBasedAgent:
        """Cria agente baseado em regras."""
        default_config = {
            "agent_type": "rule_based",
            "temp_thresholds": {
                "cooling_start": 24.0,
                "cooling_stop": 22.0,
                "heating_start": 20.0,
                "heating_stop": 22.0
            }
        }
        if config:
            default_config.update(config)
        return RuleBasedAgent(env, agent_id, default_config)

    @staticmethod
    def create_multi_agent_system(env, num_agents: Optional[int] = None,
                                config: Optional[Dict] = None) -> List[RuleBasedAgent]:
        """Cria sistema de agentes baseados em regras."""
        if num_agents is None:
            num_agents = env.num_buildings

        agents = []
        for i in range(num_agents):
            agent_config = config.copy() if config else {}
            agent_config["agent_id"] = i
            agent = RuleBasedAgentFactory.create_agent(env, i, agent_config)
            agents.append(agent)

        return agents


def create_agent_from_config(config_path: str, env, agent_id: int = 0):
    """
    Cria agente a partir de arquivo de configuração YAML.

    Args:
        config_path: Caminho do arquivo de configuração
        env: Ambiente de simulação
        agent_id: ID do agente

    Returns:
        Agente configurado
    """
    try:
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validar configuração
        AgentFactory.validate_config(config)

        # Criar agente
        agent_type = config["agents"]["type"]
        agent_config = config["agents"]

        return AgentFactory.create_agent(agent_type, env, agent_id, agent_config)

    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        raise


def create_multi_agent_from_config(config_path: str, env):
    """
    Cria sistema multi-agente a partir de arquivo de configuração YAML.

    Args:
        config_path: Caminho do arquivo de configuração
        env: Ambiente de simulação

    Returns:
        Lista de agentes configurados
    """
    try:
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validar configuração
        AgentFactory.validate_config(config)

        # Criar sistema multi-agente
        return AgentFactory.create_multi_agent_system(env, config)

    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        raise


def get_available_agent_types() -> List[str]:
    """
    Retorna lista de tipos de agentes disponíveis.

    Returns:
        List[str]: Tipos de agentes disponíveis
    """
    return ["independent", "cooperative", "centralized", "random", "rule_based"]


def get_default_config(agent_type: str) -> Dict:
    """
    Retorna configuração padrão para um tipo de agente.

    Args:
        agent_type: Tipo de agente

    Returns:
        Dict: Configuração padrão
    """
    configs = {
        "independent": {
            "agent_type": "independent",
            "learning_rate": 3e-4,
            "exploration_rate": 1.0,
            "min_exploration_rate": 0.1,
            "exploration_decay": 0.995,
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": "Tanh"
            }
        },
        "cooperative": {
            "agent_type": "cooperative",
            "learning_rate": 3e-4,
            "cooperation_strength": 0.1,
            "shared_policy": True,
            "communication_dim": 32,
            "policy_kwargs": {
                "net_arch": [256, 256, 128],
                "activation_fn": "ReLU"
            }
        },
        "centralized": {
            "agent_type": "centralized",
            "learning_rate": 3e-4,
            "coordination_strategy": "global",
            "policy_kwargs": {
                "net_arch": [512, 256, 128],
                "activation_fn": "ReLU"
            }
        },
        "random": {
            "agent_type": "random"
        },
        "rule_based": {
            "agent_type": "rule_based",
            "temp_thresholds": {
                "cooling_start": 24.0,
                "cooling_stop": 22.0,
                "heating_start": 20.0,
                "heating_stop": 22.0
            }
        }
    }

    return configs.get(agent_type, configs["independent"])


def validate_agent_config(config: Dict) -> bool:
    """
    Valida configuração de agente.

    Args:
        config: Configuração a validar

    Returns:
        bool: True se válida

    Raises:
        ValueError: Se configuração inválida
    """
    required_fields = ["agent_type"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Campo obrigatório ausente: {field}")

    # Validar tipo de agente
    valid_types = get_available_agent_types()
    if config["agent_type"] not in valid_types:
        raise ValueError(f"Tipo de agente inválido: {config['agent_type']}")

    # Validar configurações específicas
    if config["agent_type"] in ["independent", "cooperative", "centralized"]:
        if "policy_kwargs" not in config:
            config["policy_kwargs"] = {}

        if "net_arch" not in config["policy_kwargs"]:
            config["policy_kwargs"]["net_arch"] = [64, 64]

    return True


# Funções de teste para o factory
def test_agent_factory():
    """Testa funcionalidades do factory."""
    print("🧪 Testando AgentFactory...")

    try:
        from src.environment import make_citylearn_vec_env

        # Criar ambiente
        env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")

        # Testar criação de agente independente
        print("\n1. Testando IndependentAgent...")
        config = get_default_config("independent")
        agent = AgentFactory.create_agent("independent", env, 0, config)
        print(f"   ✅ Agente criado: {agent}")

        # Testar criação de agente cooperativo
        print("\n2. Testando CooperativeAgent...")
        config = get_default_config("cooperative")
        agent = AgentFactory.create_agent("cooperative", env, 1, config)
        print(f"   ✅ Agente criado: {agent}")

        # Testar criação de agente centralizado
        print("\n3. Testando CentralizedAgent...")
        config = get_default_config("centralized")
        agent = AgentFactory.create_agent("centralized", env, 2, config)
        print(f"   ✅ Agente criado: {agent}")

        # Testar sistema multi-agente
        print("\n4. Testando sistema multi-agente...")
        config = {"agent_type": "independent"}
        agents = AgentFactory.create_multi_agent_system(env, config)
        print(f"   ✅ Sistema criado: {len(agents)} agentes")

        # Testar agentes baseline
        print("\n5. Testando agentes baseline...")
        baselines = MultiAgentFactory.create_baseline_agents(env)
        print(f"   ✅ Baselines criados: {list(baselines.keys())}")

        env.close()
        print("\n✅ AgentFactory testado com sucesso!")
        return True

    except Exception as e:
        print(f"❌ Erro no teste do factory: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Executar testes do factory
    test_agent_factory()