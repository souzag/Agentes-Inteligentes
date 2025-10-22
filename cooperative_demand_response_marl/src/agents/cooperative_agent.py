#!/usr/bin/env python3
"""
Agente cooperativo para MARL no sistema de demand response.

Este módulo implementa o CooperativeAgent que considera informações de outros
agentes para tomada de decisão, implementando mecanismos de cooperação e
coordenação para otimizar o desempenho global do sistema.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class CooperativeAgent(BaseAgent):
    """
    Agente que coopera com outros agentes para otimizar objetivos globais.

    Este agente implementa mecanismos de comunicação e coordenação para
    melhorar o desempenho coletivo do sistema de demand response, considerando
    tanto objetivos individuais quanto globais.

    Args:
        env: Ambiente de simulação
        agent_id: ID único do agente
        config: Configurações do agente
        communication_protocol: Protocolo de comunicação (opcional)

    Attributes:
        comm_protocol: Protocolo de comunicação
        cooperation_strength: Força da cooperação (0.0 a 1.0)
        shared_policy: Se deve usar política compartilhada
        communication_dim: Dimensão do canal de comunicação
    """

    def __init__(self, env, agent_id: int, config: Dict, communication_protocol=None):
        """
        Inicializa o agente cooperativo.

        Args:
            env: Ambiente de simulação
            agent_id: ID único do agente
            config: Configurações do agente
            communication_protocol: Protocolo de comunicação
        """
        super().__init__(env, agent_id, config)

        # Configurações específicas do agente cooperativo
        self.comm_protocol = communication_protocol
        self.cooperation_strength = config.get("cooperation_strength", 0.1)
        self.shared_policy = config.get("shared_policy", True)
        self.communication_dim = config.get("communication_dim", 32)

        # Estado de comunicação
        self.last_messages = []
        self.communication_history = []

        # Criar política cooperativa
        self.policy = self._create_cooperative_policy()

        print(f"✅ CooperativeAgent {agent_id} inicializado com comunicação: {self.comm_protocol is not None}")

    def _create_cooperative_policy(self):
        """
        Cria política cooperativa usando Stable Baselines3.

        Returns:
            Política configurada para cooperação
        """
        try:
            from stable_baselines3 import PPO

            # Configurações da política
            policy_kwargs = self.config.get("policy_kwargs", {})
            net_arch = policy_kwargs.get("net_arch", [256, 256, 128])

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

    def select_action(self, observation: np.ndarray, messages: Optional[List] = None, **kwargs) -> np.ndarray:
        """
        Seleciona ação considerando comunicação com outros agentes.

        Args:
            observation: Observação do ambiente
            **kwargs: Argumentos adicionais (ex: mensagens)

        Returns:
            np.ndarray: Ação selecionada
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        # Receber mensagens de outros agentes (ou usar as fornecidas)
        if messages is None:
            messages = self._receive_communication()

        # Processar comunicação
        communication_state = self._process_communication(messages)

        # Selecionar ação considerando comunicação
        if communication_state is not None:
            # Combinar observação com comunicação
            combined_obs = self._combine_observation_communication(observation, communication_state)
        else:
            combined_obs = observation

        # Predizer ação usando a política
        action, _ = self.policy.predict(combined_obs, deterministic=True)

        # Validar ação
        action = self._validate_action(action)

        return action

    def update_policy(self, experience: Tuple, **kwargs) -> None:
        """
        Atualiza política baseada na experiência cooperativa.

        Args:
            experience: Tupla (obs, action, reward, next_obs, done)
            **kwargs: Argumentos adicionais
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        obs, action, reward, next_obs, done = experience

        # Processar comunicação se disponível
        messages = self._receive_communication()
        communication_state = self._process_communication(messages)

        # Combinar observações com comunicação para o buffer
        if communication_state is not None:
            combined_obs = self._combine_observation_communication(obs, communication_state)
            combined_next_obs = self._combine_observation_communication(next_obs, communication_state)
        else:
            combined_obs = obs
            combined_next_obs = next_obs

        # Adicionar experiência ao buffer da política
        if hasattr(self.policy, 'replay_buffer'):
            self.policy.replay_buffer.add(combined_obs, action, reward, combined_next_obs, done)
        else:
            # Para políticas on-policy como PPO
            self.policy.rollout_buffer.add(combined_obs, action, reward, combined_next_obs, done)

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

    def _receive_communication(self) -> List[Dict]:
        """Recebe mensagens de outros agentes."""
        if self.comm_protocol is None:
            return []

        # Receber mensagens do protocolo
        messages = self.comm_protocol.receive_messages(self.agent_id)
        self.last_messages = messages

        # Armazenar no histórico
        self.communication_history.extend(messages)
        if len(self.communication_history) > 1000:  # Manter últimos 1000
            self.communication_history = self.communication_history[-1000:]

        return messages

    def _process_communication(self, messages: List[Dict]) -> Optional[np.ndarray]:
        """
        Processa mensagens recebidas de outros agentes.

        Args:
            messages: Lista de mensagens recebidas

        Returns:
            np.ndarray: Estado de comunicação processado ou None
        """
        if not messages:
            return np.zeros(self.communication_dim)

        # Agregar informações das mensagens
        message_features = []

        for msg in messages:
            content = msg.get("content", msg)
            if isinstance(content, np.ndarray):
                # Usar diretamente se for array
                message_features.append(content.flatten())
            elif isinstance(content, (int, float)):
                # Converter escalar para array
                message_features.append(np.array([content]))
            elif isinstance(content, dict):
                # Extrair features do dicionário
                features = []
                for key, value in content.items():
                    if isinstance(value, (int, float)):
                        features.append(value)
                if features:
                    message_features.append(np.array(features))

        if message_features:
            # Concatenar e truncar/pad para o tamanho correto
            concatenated = np.concatenate(message_features)
            if len(concatenated) >= self.communication_dim:
                return concatenated[:self.communication_dim]
            else:
                # Pad com zeros
                padded = np.zeros(self.communication_dim)
                padded[:len(concatenated)] = concatenated
                return padded
        else:
            return np.zeros(self.communication_dim)

    def _combine_observation_communication(self, observation: np.ndarray,
                                         communication: np.ndarray) -> np.ndarray:
        """
        Combina observação com estado de comunicação.

        Args:
            observation: Observação do ambiente
            communication: Estado de comunicação processado

        Returns:
            np.ndarray: Observação combinada
        """
        # Concatenar observação com comunicação
        combined = np.concatenate([observation, communication])

        # Aplicar transformação cooperativa se configurada
        if self.cooperation_strength > 0:
            combined = self._apply_cooperation_transform(combined)

        return combined

    def _apply_cooperation_transform(self, combined_obs: np.ndarray) -> np.ndarray:
        """
        Aplica transformação cooperativa à observação combinada.

        Args:
            combined_obs: Observação combinada

        Returns:
            np.ndarray: Observação transformada
        """
        # Transformação simples: adicionar componente cooperativo
        # Em implementações mais avançadas, pode usar attention ou outras técnicas

        obs_dim = self.env.observation_space.shape[0]
        comm_dim = self.communication_dim

        if len(combined_obs) == obs_dim + comm_dim:
            obs_part = combined_obs[:obs_dim]
            comm_part = combined_obs[obs_dim:]

            # Garantir que comm_part tenha o mesmo shape que obs_part
            if len(comm_part) != len(obs_part):
                # Pad ou truncate para o mesmo tamanho
                if len(comm_part) < len(obs_part):
                    # Pad com zeros
                    padding = np.zeros(len(obs_part) - len(comm_part))
                    comm_part = np.concatenate([comm_part, padding])
                else:
                    # Truncate
                    comm_part = comm_part[:len(obs_part)]

            # Misturar observação com comunicação
            mixed = obs_part + self.cooperation_strength * comm_part

            return mixed
        else:
            return combined_obs

    def send_communication(self, content: Any, receiver_id: str = "all") -> None:
        """
        Envia mensagem para outros agentes.

        Args:
            content: Conteúdo da mensagem
            receiver_id: ID do destinatário ("all" para todos)
        """
        if self.comm_protocol is None:
            return

        # Preparar mensagem
        message = {
            "sender": self.agent_id,
            "receiver": receiver_id,
            "content": content,
            "timestamp": np.random.randint(0, 1000)  # Timestamp simples
        }

        # Enviar através do protocolo
        if receiver_id == "all":
            # Enviar para todos os outros agentes
            for i in range(self.env.num_buildings):
                if i != self.agent_id:
                    self.comm_protocol.send_message(self.agent_id, i, content)
        else:
            self.comm_protocol.send_message(self.agent_id, receiver_id, content)

    def get_cooperation_metrics(self) -> Dict:
        """
        Retorna métricas de cooperação do agente.

        Returns:
            Dict: Métricas de cooperação
        """
        return {
            "agent_id": self.agent_id,
            "communication_enabled": self.comm_protocol is not None,
            "cooperation_strength": self.cooperation_strength,
            "messages_received": len(self.last_messages),
            "communication_history_size": len(self.communication_history),
            "shared_policy": self.shared_policy
        }

    def train(self, total_timesteps: int, eval_freq: int = 1000, **kwargs) -> None:
        """
        Treina o agente cooperativo por um número de passos.

        Args:
            total_timesteps: Número total de passos de treinamento
            eval_freq: Frequência de avaliação
            **kwargs: Argumentos adicionais para o treinamento
        """
        if self.policy is None:
            raise RuntimeError("Política não inicializada")

        print(f"🏋️ Treinando CooperativeAgent {self.agent_id} por {total_timesteps} passos...")

        # Configurar callback para logging
        from stable_baselines3.common.callbacks import BaseCallback

        class CooperativeTrainingCallback(BaseCallback):
            def __init__(self, agent, eval_freq=1000):
                super().__init__()
                self.agent = agent
                self.eval_freq = eval_freq

            def _on_step(self):
                if self.n_calls % self.eval_freq == 0:
                    stats = self.agent.get_training_stats()
                    coop_metrics = self.agent.get_cooperation_metrics()
                    print(f"   Passo {self.n_calls}: Recompensa = {stats['mean_reward']:.3f}, "
                          f"Mensagens = {coop_metrics['messages_received']}")
                return True

        callback = CooperativeTrainingCallback(self, eval_freq)

        # Treinar política
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )

        print(f"✅ CooperativeAgent {self.agent_id} treinado!")

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Avalia performance do agente cooperativo.

        Args:
            num_episodes: Número de episódios para avaliação

        Returns:
            Dict: Métricas de performance
        """
        print(f"📊 Avaliando CooperativeAgent {self.agent_id} por {num_episodes} episódios...")

        episode_rewards = []
        episode_lengths = []
        communication_counts = []

        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            messages_count = 0
            done = False

            while not done:
                # Receber comunicação
                messages = self._receive_communication()
                messages_count += len(messages)

                # Selecionar ação
                action = self.select_action(obs)

                # Enviar comunicação
                self.send_communication(obs)

                # Executar ação
                obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                # Limitar comprimento do episódio
                if episode_length > 10000:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            communication_counts.append(messages_count)

        # Calcular estatísticas
        rewards = np.array(episode_rewards)
        lengths = np.array(episode_lengths)
        comm_counts = np.array(communication_counts)

        results = {
            "agent_id": self.agent_id,
            "agent_type": "CooperativeAgent",
            "num_episodes": num_episodes,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "mean_communication": np.mean(comm_counts),
            "total_communication": np.sum(comm_counts),
            "total_timesteps": self.total_timesteps
        }

        print(f"   - Recompensa média: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"   - Comunicação média: {results['mean_communication']:.1f} mensagens/episódio")

        return results

    def save_model(self, filepath: str) -> None:
        """
        Salva modelo do agente cooperativo.

        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.policy and hasattr(self.policy, 'save'):
            self.policy.save(filepath)
            print(f"💾 Modelo do CooperativeAgent {self.agent_id} salvo em {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Carrega modelo do agente cooperativo.

        Args:
            filepath: Caminho do modelo a carregar
        """
        if self.policy and hasattr(self.policy, 'load'):
            self.policy.load(filepath)
            print(f"📂 Modelo do CooperativeAgent {self.agent_id} carregado de {filepath}")

    def get_policy_info(self) -> Dict:
        """
        Retorna informações da política cooperativa.

        Returns:
            Dict: Informações da política
        """
        info = super().get_policy_info()

        # Adicionar informações específicas de cooperação
        info.update({
            "cooperation_strength": self.cooperation_strength,
            "communication_dim": self.communication_dim,
            "shared_policy": self.shared_policy,
            "communication_protocol": type(self.comm_protocol).__name__ if self.comm_protocol else None,
            "messages_in_buffer": len(self.communication_buffer),
            "communication_history_size": len(self.communication_history)
        })

        return info


class CooperativeAgentFactory:
    """
    Factory específico para agentes cooperativos.

    Permite configuração otimizada para agentes cooperativos
    e facilita a criação de sistemas multi-agente cooperativos.
    """

    @staticmethod
    def create_agent(env, agent_id: int, config: Optional[Dict] = None,
                    communication_protocol=None) -> CooperativeAgent:
        """
        Cria agente cooperativo com configuração padrão.

        Args:
            env: Ambiente de simulação
            agent_id: ID do agente
            config: Configurações específicas (opcional)
            communication_protocol: Protocolo de comunicação

        Returns:
            CooperativeAgent: Agente configurado
        """
        default_config = {
            "agent_type": "cooperative",
            "learning_rate": 3e-4,
            "cooperation_strength": 0.1,
            "shared_policy": True,
            "communication_dim": 32,
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
                "net_arch": [256, 256, 128],
                "activation_fn": "ReLU"
            }
        }

        if config:
            default_config.update(config)

        return CooperativeAgent(env, agent_id, default_config, communication_protocol)

    @staticmethod
    def create_multi_agent_system(env, communication_protocol=None,
                                config: Optional[Dict] = None) -> List[CooperativeAgent]:
        """
        Cria sistema de múltiplos agentes cooperativos.

        Args:
            env: Ambiente de simulação
            communication_protocol: Protocolo de comunicação
            config: Configurações base para todos os agentes

        Returns:
            List[CooperativeAgent]: Lista de agentes cooperativos
        """
        num_agents = env.num_buildings

        agents = []
        for i in range(num_agents):
            agent_config = config.copy() if config else {}
            agent_config["agent_id"] = i

            agent = CooperativeAgentFactory.create_agent(env, i, agent_config, communication_protocol)
            agents.append(agent)

        print(f"✅ Sistema de {len(agents)} agentes cooperativos criado")
        return agents