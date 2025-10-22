#!/usr/bin/env python3
"""
Agente centralizado para MARL no sistema de demand response.

Este módulo implementa o CentralizedAgent que controla todos os prédios
de forma centralizada, otimizando o desempenho global do sistema através
de uma política central unificada.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class CentralizedAgent(BaseAgent):
    """
    Agente centralizado que controla todos os prédios simultaneamente.

    Este agente implementa uma política central que recebe o estado global
    de todos os prédios e gera ações para todos eles simultaneamente,
    otimizando o desempenho do sistema como um todo.

    Args:
        env: Ambiente de simulação
        agent_id: ID único do agente (geralmente 0 para centralizado)
        config: Configurações do agente

    Attributes:
        num_buildings: Número de prédios controlados
        global_policy: Política centralizada
        coordination_strategy: Estratégia de coordenação
    """

    def __init__(self, env, agent_id: int, config: Dict):
        """
        Inicializa o agente centralizado.

        Args:
            env: Ambiente de simulação
            agent_id: ID único do agente
            config: Configurações do agente
        """
        super().__init__(env, agent_id, config)

        # Configurações específicas do agente centralizado
        self.num_buildings = env.num_buildings
        self.coordination_strategy = config.get("coordination_strategy", "global")

        # Criar política centralizada
        self.policy = self._create_centralized_policy()

        print(f"✅ CentralizedAgent {agent_id} inicializado para controlar {self.num_buildings} prédios")

    def _create_centralized_policy(self):
        """
        Cria política centralizada usando Stable Baselines3.

        Returns:
            Política configurada para controle centralizado
        """
        try:
            from stable_baselines3 import PPO

            # Configurações da política
            policy_kwargs = self.config.get("policy_kwargs", {})
            net_arch = policy_kwargs.get("net_arch", [512, 256, 128])

            # Usar política padrão do SB3 para evitar problemas de configuração
            policy = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.learning_rate,
                n_steps=self.config.get("n_steps", 2048),
                batch_size=self.config.get("batch_size", 128),  # Batch maior para centralizado
                n_epochs=self.config.get("n_epochs", 10),
                gamma=self.gamma,
                gae_lambda=self.config.get("gae_lambda", 0.95),
                clip_range=self.config.get("clip_range", 0.2),
                ent_coef=self.entropy_coef,
                vf_coef=self.config.get("vf_coef", 0.5),
                max_grad_norm=self.config.get("max_grad_norm", 0.5),
                verbose=0,
                seed=self.config.get("seed", None)
            )

            return policy

        except ImportError:
            raise ImportError("Stable Baselines3 não encontrado. Instale com: pip install stable-baselines3")

    def select_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Seleciona ações para todos os prédios simultaneamente.

        Args:
            observation: Observação global (estado de todos os prédios)
            **kwargs: Argumentos adicionais (ignorados para centralizado)

        Returns:
            np.ndarray: Ações para todos os prédios
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        # A observação já contém estado de todos os prédios
        # Predizer ações para todos os prédios
        actions, _ = self.policy.predict(observation, deterministic=True)

        # Validar ações
        actions = self._validate_action(actions)

        return actions

    def update_policy(self, experience: Tuple, **kwargs) -> None:
        """
        Atualiza política centralizada baseada na experiência global.

        Args:
            experience: Tupla (obs, action, reward, next_obs, done)
            **kwargs: Argumentos adicionais
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        obs, action, reward, next_obs, done = experience

        # Para agente centralizado, a recompensa é global (soma de todas)
        if isinstance(reward, np.ndarray):
            global_reward = np.sum(reward)
        else:
            global_reward = reward

        # Adicionar experiência ao buffer da política
        if hasattr(self.policy, 'replay_buffer'):
            self.policy.replay_buffer.add(obs, action, global_reward, next_obs, done)
        else:
            # Para políticas on-policy como PPO
            self.policy.rollout_buffer.add(obs, action, global_reward, next_obs, done)

        # Atualizar política se necessário
        if hasattr(self.policy, 'collect_rollouts'):
            if self.policy.num_timesteps >= self.policy.n_steps:
                self.policy.train()
        else:
            if len(self.policy.replay_buffer) >= self.policy.batch_size:
                self.policy.train()

        # Log do passo de treinamento
        loss = getattr(self.policy, 'loss', None)
        self._log_training_step(experience, loss)

    def coordinate_actions(self, building_states: List[np.ndarray]) -> np.ndarray:
        """
        Coordena ações baseado nos estados individuais dos prédios.

        Args:
            building_states: Lista de estados de cada prédio

        Returns:
            np.ndarray: Ações coordenadas para todos os prédios
        """
        if self.coordination_strategy == "global":
            return self._global_coordination(building_states)
        elif self.coordination_strategy == "hierarchical":
            return self._hierarchical_coordination(building_states)
        elif self.coordination_strategy == "priority":
            return self._priority_coordination(building_states)
        else:
            # Fallback para coordenação global
            return self._global_coordination(building_states)

    def _global_coordination(self, building_states: List[np.ndarray]) -> np.ndarray:
        """
        Coordenação global: otimizar todos os prédios simultaneamente.

        Args:
            building_states: Estados de todos os prédios

        Returns:
            np.ndarray: Ações para todos os prédios
        """
        # Concatenar todos os estados
        global_state = np.concatenate(building_states)

        # Selecionar ações usando a política centralizada
        return self.select_action(global_state)

    def _hierarchical_coordination(self, building_states: List[np.ndarray]) -> np.ndarray:
        """
        Coordenação hierárquica: agrupar prédios e otimizar por grupos.

        Args:
            building_states: Estados de todos os prédios

        Returns:
            np.ndarray: Ações para todos os prédios
        """
        # Dividir prédios em grupos (simplificado)
        group_size = max(1, self.num_buildings // 3)
        actions = []

        for i in range(0, self.num_buildings, group_size):
            group_states = building_states[i:i+group_size]
            group_global_state = np.concatenate(group_states)

            # Otimizar grupo
            group_actions = self.select_action(group_global_state)
            actions.extend(group_actions)

        return np.array(actions[:self.num_buildings])

    def _priority_coordination(self, building_states: List[np.ndarray]) -> np.ndarray:
        """
        Coordenação por prioridade: otimizar prédios críticos primeiro.

        Args:
            building_states: Estados de todos os prédios

        Returns:
            np.ndarray: Ações para todos os prédios
        """
        # Calcular prioridades baseado no estado (simplificado)
        priorities = []
        for i, state in enumerate(building_states):
            # Prioridade baseada na temperatura (simplificado)
            if len(state) >= 7:
                temp_deviation = abs(state[3] - 23.0)  # Desvio da temperatura ideal
                priority = temp_deviation
            else:
                priority = 0.0
            priorities.append((i, priority))

        # Ordenar por prioridade (maior primeiro)
        priorities.sort(key=lambda x: x[1], reverse=True)

        # Otimizar na ordem de prioridade
        actions = [0.0] * self.num_buildings
        remaining_capacity = np.ones(self.num_buildings)  # Capacidade disponível

        for building_idx, priority in priorities:
            # Otimizar prédio atual considerando restrições
            building_state = building_states[building_idx]

            # Ação considerando capacidade restante
            action = self.select_action(building_state)[0]
            action = np.clip(action, -remaining_capacity[building_idx], remaining_capacity[building_idx])

            actions[building_idx] = action

            # Atualizar capacidade restante
            remaining_capacity = np.maximum(remaining_capacity - abs(action), 0)

        return np.array(actions)

    def train(self, total_timesteps: int, eval_freq: int = 1000, **kwargs) -> None:
        """
        Treina o agente centralizado por um número de passos.

        Args:
            total_timesteps: Número total de passos de treinamento
            eval_freq: Frequência de avaliação
            **kwargs: Argumentos adicionais para o treinamento
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        print(f"🏛️ Treinando CentralizedAgent {self.agent_id} para {self.num_buildings} prédios...")
        print(f"   Coordenação: {self.coordination_strategy}")

        # Configurar callback para logging
        from stable_baselines3.common.callbacks import BaseCallback

        class CentralizedTrainingCallback(BaseCallback):
            def __init__(self, agent, eval_freq=1000):
                super().__init__()
                self.agent = agent
                self.eval_freq = eval_freq

            def _on_step(self):
                if self.n_calls % self.eval_freq == 0:
                    stats = self.agent.get_training_stats()
                    print(f"   Passo {self.n_calls}: Recompensa global = {stats['mean_reward']:.3f}")
                return True

        callback = CentralizedTrainingCallback(self, eval_freq)

        # Treinar política
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )

        print(f"✅ CentralizedAgent {self.agent_id} treinado!")

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Avalia performance do agente centralizado.

        Args:
            num_episodes: Número de episódios para avaliação

        Returns:
            Dict: Métricas de performance
        """
        print(f"📊 Avaliando CentralizedAgent {self.agent_id} por {num_episodes} episódios...")

        episode_rewards = []
        episode_lengths = []
        coordination_metrics = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Selecionar ações para todos os prédios
                actions = self.select_action(obs)

                # Executar ações
                obs, reward, done, info = self.env.step(actions)

                episode_reward += reward
                episode_length += 1

                # Limitar comprimento do episódio
                if episode_length > 10000:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Calcular métricas de coordenação
            coordination_metric = self._calculate_coordination_metric()
            coordination_metrics.append(coordination_metric)

        # Calcular estatísticas
        rewards = np.array(episode_rewards)
        lengths = np.array(episode_lengths)
        coord_metrics = np.array(coordination_metrics)

        results = {
            "agent_id": self.agent_id,
            "agent_type": "CentralizedAgent",
            "num_episodes": num_episodes,
            "num_buildings": self.num_buildings,
            "coordination_strategy": self.coordination_strategy,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "mean_coordination": np.mean(coord_metrics),
            "total_timesteps": self.total_timesteps
        }

        print(f"   - Recompensa média: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"   - Coordenação média: {results['mean_coordination']:.3f}")

        return results

    def _calculate_coordination_metric(self) -> float:
        """
        Calcula métrica de coordenação das ações.

        Returns:
            float: Métrica de coordenação (0.0 a 1.0)
        """
        # Métrica simples baseada na variabilidade das ações
        # Em implementações mais avançadas, pode usar métricas mais sofisticadas

        if not hasattr(self, 'last_actions'):
            return 0.0

        if len(self.last_actions) < 2:
            return 1.0

        # Calcular correlação das ações como métrica de coordenação
        try:
            correlation = np.corrcoef(self.last_actions, self.last_actions)[0, 1]
            return max(0.0, correlation)  # Retornar apenas valores positivos
        except:
            return 0.0

    def save_model(self, filepath: str) -> None:
        """
        Salva modelo do agente centralizado.

        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.policy and hasattr(self.policy, 'save'):
            self.policy.save(filepath)
            print(f"💾 Modelo do CentralizedAgent {self.agent_id} salvo em {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Carrega modelo do agente centralizado.

        Args:
            filepath: Caminho do modelo a carregar
        """
        if self.policy and hasattr(self.policy, 'load'):
            self.policy.load(filepath)
            print(f"📂 Modelo do CentralizedAgent {self.agent_id} carregado de {filepath}")

    def get_policy_info(self) -> Dict:
        """
        Retorna informações da política centralizada.

        Returns:
            Dict: Informações da política
        """
        info = super().get_policy_info()

        # Adicionar informações específicas de centralização
        info.update({
            "num_buildings_controlled": self.num_buildings,
            "coordination_strategy": self.coordination_strategy,
            "centralized_control": True
        })

        return info

    def get_global_state(self) -> Dict:
        """
        Retorna estado global do sistema.

        Returns:
            Dict: Estado global
        """
        return {
            "num_buildings": self.num_buildings,
            "coordination_strategy": self.coordination_strategy,
            "total_timesteps": self.total_timesteps,
            "training_history_size": len(self.training_history)
        }


class CentralizedAgentFactory:
    """
    Factory específico para agentes centralizados.

    Permite configuração otimizada para agentes centralizados
    e facilita a criação de sistemas de controle centralizado.
    """

    @staticmethod
    def create_agent(env, agent_id: int = 0, config: Optional[Dict] = None) -> CentralizedAgent:
        """
        Cria agente centralizado com configuração padrão.

        Args:
            env: Ambiente de simulação
            agent_id: ID do agente (geralmente 0)
            config: Configurações específicas (opcional)

        Returns:
            CentralizedAgent: Agente configurado
        """
        default_config = {
            "agent_type": "centralized",
            "learning_rate": 3e-4,
            "coordination_strategy": "global",
            "n_steps": 2048,
            "batch_size": 128,  # Batch maior para centralizado
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [512, 256, 128],
                "activation_fn": "ReLU"
            }
        }

        if config:
            default_config.update(config)

        return CentralizedAgent(env, agent_id, default_config)

    @staticmethod
    def create_centralized_system(env, config: Optional[Dict] = None) -> CentralizedAgent:
        """
        Cria sistema centralizado com um único agente controlador.

        Args:
            env: Ambiente de simulação
            config: Configurações do agente centralizado

        Returns:
            CentralizedAgent: Agente centralizado configurado
        """
        agent_config = config.copy() if config else {}

        agent = CentralizedAgentFactory.create_agent(env, agent_id=0, config=agent_config)

        print(f"✅ Sistema centralizado criado: 1 agente controlando {env.num_buildings} prédios")
        return agent