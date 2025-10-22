#!/usr/bin/env python3
"""
Agente independente para MARL no sistema de demand response.

Este módulo implementa o IndependentAgent que aprende políticas individuais
sem considerar ou cooperar com outros agentes. Serve como baseline para
comparação com agentes cooperativos.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class IndependentAgent(BaseAgent):
    """
    Agente que aprende independentemente sem cooperação.

    Este agente implementa uma política individual usando Stable Baselines3
    e não considera informações de outros agentes. É útil como baseline
    para comparar com abordagens cooperativas.

    Args:
        env: Ambiente de simulação
        agent_id: ID único do agente
        config: Configurações do agente

    Attributes:
        policy: Política de aprendizado individual
        exploration_rate: Taxa de exploração atual
        learning_rate: Taxa de aprendizado
    """

    def __init__(self, env, agent_id: int, config: Dict):
        """
        Inicializa o agente independente.

        Args:
            env: Ambiente de simulação
            agent_id: ID único do agente
            config: Configurações do agente
        """
        super().__init__(env, agent_id, config)

        # Configurações específicas do agente independente
        self.exploration_rate = config.get("exploration_rate", 1.0)
        self.min_exploration_rate = config.get("min_exploration_rate", 0.1)
        self.exploration_decay = config.get("exploration_decay", 0.995)

        # Criar política individual
        self.policy = self._create_policy()

        print(f"✅ IndependentAgent {agent_id} inicializado com política {type(self.policy).__name__}")

    def _create_policy(self):
        """
        Cria política individual usando Stable Baselines3.

        Returns:
            Política configurada para o agente
        """
        try:
            from stable_baselines3 import PPO

            # Configurações da política
            policy_kwargs = self.config.get("policy_kwargs", {})
            net_arch = policy_kwargs.get("net_arch", [64, 64])

            # Usar política padrão do SB3 para evitar problemas de configuração
            policy = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.learning_rate,
                n_steps=self.config.get("n_steps", 2048),
                batch_size=self.config.get("batch_size", 64),
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
        Seleciona ação baseada na observação.

        Args:
            observation: Observação do ambiente
            **kwargs: Argumentos adicionais (ignorados para agente independente)

        Returns:
            np.ndarray: Ação selecionada
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        # Predizer ação usando a política
        action, _ = self.policy.predict(observation, deterministic=True)

        # Adicionar exploração se configurada
        if self.exploration_rate > 0:
            noise = np.random.normal(0, self.exploration_rate, size=action.shape)
            action = action + noise

        # Validar ação
        action = self._validate_action(action)

        return action

    def update_policy(self, experience: Tuple, **kwargs) -> None:
        """
        Atualiza política baseada na experiência.

        Args:
            experience: Tupla (obs, action, reward, next_obs, done)
            **kwargs: Argumentos adicionais
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        obs, action, reward, next_obs, done = experience

        # Adicionar experiência ao buffer da política
        if hasattr(self.policy, 'replay_buffer'):
            self.policy.replay_buffer.add(obs, action, reward, next_obs, done)
        else:
            # Para políticas on-policy como PPO, adicionar ao rollout buffer
            self.policy.rollout_buffer.add(obs, action, reward, next_obs, done)

        # Atualizar política se necessário
        if hasattr(self.policy, 'collect_rollouts'):
            # PPO e políticas similares
            if self.policy.num_timesteps >= self.policy.n_steps:
                self.policy.train()
        else:
            # Políticas off-policy
            if len(self.policy.replay_buffer) >= self.policy.batch_size:
                self.policy.train()

        # Atualizar taxa de exploração
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

        # Log do passo de treinamento
        loss = getattr(self.policy, 'loss', None)
        self._log_training_step(experience, loss)

    def train(self, total_timesteps: int, eval_freq: int = 1000, **kwargs) -> None:
        """
        Treina o agente por um número de passos.

        Args:
            total_timesteps: Número total de passos de treinamento
            eval_freq: Frequência de avaliação
            **kwargs: Argumentos adicionais para o treinamento
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        print(f"🏋️ Treinando IndependentAgent {self.agent_id} por {total_timesteps} passos...")

        # Configurar callback para logging
        from stable_baselines3.common.callbacks import BaseCallback

        class TrainingCallback(BaseCallback):
            def __init__(self, agent, eval_freq=1000):
                super().__init__()
                self.agent = agent
                self.eval_freq = eval_freq

            def _on_step(self):
                if self.n_calls % self.eval_freq == 0:
                    stats = self.agent.get_training_stats()
                    print(f"   Passo {self.n_calls}: Recompensa média = {stats['mean_reward']:.3f}")
                return True

        callback = TrainingCallback(self, eval_freq)

        # Treinar política
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )

        print(f"✅ IndependentAgent {self.agent_id} treinado!")

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Avalia performance do agente.

        Args:
            num_episodes: Número de episódios para avaliação

        Returns:
            Dict: Métricas de performance
        """
        print(f"📊 Avaliando IndependentAgent {self.agent_id} por {num_episodes} episódios...")

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

                # Limitar comprimento do episódio para evitar loops infinitos
                if episode_length > 10000:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        # Calcular estatísticas
        rewards = np.array(episode_rewards)
        lengths = np.array(episode_lengths)

        results = {
            "agent_id": self.agent_id,
            "agent_type": "IndependentAgent",
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

    def save_model(self, filepath: str) -> None:
        """
        Salva modelo do agente.

        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.policy and hasattr(self.policy, 'save'):
            self.policy.save(filepath)
            print(f"💾 Modelo do IndependentAgent {self.agent_id} salvo em {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Carrega modelo do agente.

        Args:
            filepath: Caminho do modelo a carregar
        """
        if self.policy and hasattr(self.policy, 'load'):
            self.policy.load(filepath)
            print(f"📂 Modelo do IndependentAgent {self.agent_id} carregado de {filepath}")

    def get_policy_info(self) -> Dict:
        """
        Retorna informações da política.

        Returns:
            Dict: Informações da política
        """
        if self.policy is None:
            return {"policy_type": None, "parameters": 0}

        info = {
            "policy_type": type(self.policy).__name__,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "total_timesteps": self.total_timesteps
        }

        # Adicionar informações específicas do SB3 se disponíveis
        if hasattr(self.policy, 'policy'):
            policy_net = self.policy.policy
            if hasattr(policy_net, 'parameters'):
                total_params = sum(p.numel() for p in policy_net.parameters())
                info["parameters"] = total_params

        return info

    def reset_exploration(self, exploration_rate: float = 1.0) -> None:
        """
        Reseta taxa de exploração.

        Args:
            exploration_rate: Nova taxa de exploração
        """
        self.exploration_rate = exploration_rate
        print(f"🔄 Exploração do IndependentAgent {self.agent_id} resetada para {exploration_rate}")


class IndependentAgentFactory:
    """
    Factory específico para agentes independentes.

    Permite configuração otimizada para agentes independentes
    e facilita a criação de múltiplos agentes similares.
    """

    @staticmethod
    def create_agent(env, agent_id: int, config: Optional[Dict] = None) -> IndependentAgent:
        """
        Cria agente independente com configuração padrão.

        Args:
            env: Ambiente de simulação
            agent_id: ID do agente
            config: Configurações específicas (opcional)

        Returns:
            IndependentAgent: Agente configurado
        """
        default_config = {
            "agent_type": "independent",
            "learning_rate": 3e-4,
            "exploration_rate": 1.0,
            "min_exploration_rate": 0.1,
            "exploration_decay": 0.995,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": "Tanh"
            }
        }

        if config:
            default_config.update(config)

        return IndependentAgent(env, agent_id, default_config)

    @staticmethod
    def create_multi_agent_system(env, num_agents: Optional[int] = None,
                                config: Optional[Dict] = None) -> List[IndependentAgent]:
        """
        Cria sistema de múltiplos agentes independentes.

        Args:
            env: Ambiente de simulação
            num_agents: Número de agentes (se None, usa env.num_buildings)
            config: Configurações base para todos os agentes

        Returns:
            List[IndependentAgent]: Lista de agentes independentes
        """
        if num_agents is None:
            num_agents = env.num_buildings

        agents = []
        for i in range(num_agents):
            agent_config = config.copy() if config else {}
            agent_config["agent_id"] = i

            agent = IndependentAgentFactory.create_agent(env, i, agent_config)
            agents.append(agent)

        print(f"✅ Sistema de {len(agents)} agentes independentes criado")
        return agents