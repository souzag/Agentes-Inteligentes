#!/usr/bin/env python3
"""
Ambiente vetorizado customizado para CityLearn com suporte a MARL cooperativo.

Este módulo implementa a classe CityLearnVecEnv que permite o treinamento de
algoritmos multi-agente cooperativos usando Stable Baselines3.
"""

import os
import numpy as np
import gymnasium
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class CityLearnVecEnv(gymnasium.Env):
    """
    Ambiente vetorizado multi-agente para CityLearn.

    Esta classe implementa um wrapper customizado para o ambiente CityLearn,
    otimizado para treinamento de algoritmos MARL (Multi-Agent Reinforcement Learning)
    cooperativos usando Stable Baselines3.

    Args:
        dataset_name (str): Nome do dataset CityLearn
        reward_function (str): Tipo de função de recompensa ("local", "global", "cooperative")
        communication (bool): Ativar comunicação entre agentes
        normalize_obs (bool): Normalizar observações
        max_episode_length (int): Comprimento máximo do episódio
        seed (int): Seed para reprodutibilidade

    Attributes:
        num_buildings (int): Número de prédios/agentes
        observation_space (gymnasium.spaces.Box): Espaço de observação
        action_space (gymnasium.spaces.Box): Espaço de ação
        citylearn_env: Ambiente CityLearn base
        reward_function (callable): Função de recompensa
        communication_enabled (bool): Flag de comunicação
    """

    def __init__(
        self,
        dataset_name: str = "citylearn_challenge_2022_phase_1",
        reward_function: str = "cooperative",
        communication: bool = True,
        normalize_obs: bool = True,
        max_episode_length: int = 8760,
        seed: Optional[int] = None
    ):
        super().__init__()

        # Configurações básicas
        self.dataset_name = dataset_name
        self.reward_function_type = reward_function
        self.communication_enabled = communication
        self.normalize_obs = normalize_obs
        self.max_episode_length = max_episode_length
        self.seed = seed

        # Carregar ambiente CityLearn
        self._load_citylearn_env()

        # Configurar espaços
        self._setup_spaces()

        # Configurar função de recompensa
        self._setup_reward_function()

        # Configurar comunicação
        if self.communication_enabled:
            self._setup_communication()

        # Estatísticas para normalização
        self.obs_mean = None
        self.obs_std = None
        self.reward_mean = None
        self.reward_std = None

        # Estado do episódio
        self.episode_step = 0
        self.episode_reward = 0.0

        print(f"✅ CityLearnVecEnv inicializado:")
        print(f"   - Dataset: {self.dataset_name}")
        print(f"   - Prédios: {self.num_buildings}")
        print(f"   - Observation space: {self.observation_space.shape}")
        print(f"   - Action space: {self.action_space.shape}")
        print(f"   - Reward function: {self.reward_function_type}")
        print(f"   - Communication: {self.communication_enabled}")

    def _load_citylearn_env(self):
        """Carrega o ambiente CityLearn base."""
        try:
            from citylearn.citylearn import DataSet, CityLearnEnv

            # Carregar dataset
            dataset = DataSet(self.dataset_name)
            dataset_path = dataset.get_dataset(self.dataset_name)

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

            # Criar ambiente
            self.citylearn_env = CityLearnEnv(dataset_path)

            # Configurar seed se fornecido
            if self.seed is not None:
                self.citylearn_env.reset(seed=self.seed)

            # Obter número de prédios
            self.num_buildings = len(self.citylearn_env.buildings)

        except ImportError:
            raise ImportError(
                "CityLearn não encontrado. Instale com: pip install citylearn"
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar CityLearn: {e}")

    def _setup_spaces(self):
        """Configura os espaços de observação e ação."""
        # Observation space: concatenação de todos os prédios
        if self.num_buildings > 0:
            # Usar limites mais permissivos baseados nos espaços originais
            first_building = self.citylearn_env.buildings[0]
            obs_low = first_building.observation_space.low
            obs_high = first_building.observation_space.high
            obs_shape = first_building.observation_space.shape

            # Verificar se todos os prédios têm o mesmo formato
            for building in self.citylearn_env.buildings[1:]:
                if building.observation_space.shape != obs_shape:
                    raise ValueError("Prédios têm formatos de observação diferentes")

            # Espaço de observação vetorizado com limites mais permissivos
            obs_dim = self.num_buildings * obs_shape[0]
            self.observation_space = gymnasium.spaces.Box(
                low=np.concatenate([obs_low] * self.num_buildings),
                high=np.concatenate([obs_high] * self.num_buildings),
                shape=(obs_dim,),
                dtype=np.float32
            )

            # Espaço de ação: uma ação por prédio
            action_low = first_building.action_space.low[0]
            action_high = first_building.action_space.high[0]

            self.action_space = gymnasium.spaces.Box(
                low=np.array([action_low] * self.num_buildings),
                high=np.array([action_high] * self.num_buildings),
                shape=(self.num_buildings,),
                dtype=np.float32
            )
        else:
            raise ValueError("Nenhum prédio encontrado no ambiente")

    def _setup_reward_function(self):
        """Configura a função de recompensa."""
        if self.reward_function_type == "cooperative":
            self.reward_function = self._cooperative_reward
        elif self.reward_function_type == "global":
            self.reward_function = self._global_reward
        elif self.reward_function_type == "local":
            self.reward_function = self._local_reward
        else:
            raise ValueError(f"Tipo de recompensa inválido: {self.reward_function_type}")

    def _setup_communication(self):
        """Configura o sistema de comunicação."""
        # Por enquanto, implementamos comunicação simples através do estado global
        # Futuramente pode ser expandido para protocolos mais complexos
        self.communication_buffer = {}

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reseta o ambiente para um novo episódio.

        Returns:
            tuple: (observations, info)
                observations: Observações iniciais vetorizadas
                info: Informações adicionais
        """
        # Reset do ambiente CityLearn
        citylearn_result = self.citylearn_env.reset(**kwargs)

        if isinstance(citylearn_result, tuple):
            obs, info = citylearn_result
        else:
            obs, info = citylearn_result, {}

        # Converter para formato vetorizado
        vectorized_obs = self._concatenate_observations(obs)

        # Aplicar normalização se habilitada
        if self.normalize_obs:
            vectorized_obs = self._normalize_observations(vectorized_obs)

        # Reset estado do episódio
        self.episode_step = 0
        self.episode_reward = 0.0

        return vectorized_obs, info

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Union[bool, np.ndarray], Dict]:
        """
        Executa um passo no ambiente.

        Args:
            actions: Ações para todos os prédios (shape: num_buildings,)

        Returns:
            tuple: (observations, rewards, dones, infos)
        """
        # Validar ações
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.float32)
        else:
            actions = np.array(actions, dtype=np.float32)
        if actions.shape != (self.num_buildings,):
            raise ValueError(f"Forma das ações incorreta: {actions.shape}, esperado: ({self.num_buildings},)")

        # Clipping das ações se necessário
        actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Converter ações para formato do CityLearn
        building_actions = self._split_actions(actions)

        # Executar passo no CityLearn
        citylearn_result = self.citylearn_env.step(building_actions)

        # Processar resultado
        if len(citylearn_result) == 4:
            obs, rewards, done, info = citylearn_result
        else:
            obs, rewards, done, truncated, info = citylearn_result
            # Combinar done e truncated para compatibilidade
            done = done or truncated

        # Converter para formato vetorizado
        vectorized_obs = self._concatenate_observations(obs)

        # Aplicar função de recompensa customizada
        if self.reward_function:
            rewards = self.reward_function(obs, building_actions, info)

        # Aplicar normalização se habilitada
        if self.normalize_obs:
            vectorized_obs = self._normalize_observations(vectorized_obs)

        # Atualizar estado do episódio
        self.episode_step += 1
        self.episode_reward += np.sum(rewards)

        # Verificar fim do episódio
        if self.episode_step >= self.max_episode_length:
            done = True

        # Adicionar log para debugging
        if self.episode_step % 1000 == 0:
            print(f"   - Passo {self.episode_step}: Done={done}, Recompensas={np.mean(rewards):.3f}")

        return vectorized_obs, rewards, done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Renderiza o estado do ambiente.

        Args:
            mode: Modo de renderização ("human", "rgb_array")

        Returns:
            Imagem renderizada se mode="rgb_array"
        """
        # Por enquanto, delega para o CityLearn
        return self.citylearn_env.render(mode=mode)

    def close(self):
        """Fecha o ambiente e libera recursos."""
        if hasattr(self.citylearn_env, 'close'):
            self.citylearn_env.close()

    # Funções auxiliares privadas

    def _concatenate_observations(self, observations) -> np.ndarray:
        """Concatena observações de todos os prédios."""
        if isinstance(observations, list):
            # Lista de arrays - um por prédio
            concatenated = np.concatenate([np.array(obs, dtype=np.float32) for obs in observations])
        else:
            # Array único - converter para formato correto
            concatenated = np.array(observations, dtype=np.float32).flatten()

        return concatenated

    def _split_actions(self, actions: np.ndarray) -> List:
        """Divide ações vetorizadas em ações individuais por prédio."""
        # CityLearn espera ações no formato [[action1], [action2], ...]
        return [[actions[i]] for i in range(self.num_buildings)]

    def _normalize_observations(self, observations: np.ndarray) -> np.ndarray:
        """Normaliza observações usando estatísticas pré-computadas."""
        if self.obs_mean is None or self.obs_std is None:
            # Inicializar estatísticas se não existirem
            self.obs_mean = np.zeros_like(observations)
            self.obs_std = np.ones_like(observations)

        # Evitar divisão por zero
        normalized = (observations - self.obs_mean) / (self.obs_std + 1e-8)
        return normalized

    # Funções de recompensa

    def _local_reward(self, observations, actions, info) -> np.ndarray:
        """Calcula recompensa baseada apenas no estado individual."""
        # Por enquanto, retorna as recompensas originais do CityLearn
        # Futuramente pode ser customizada
        if hasattr(info, 'get') and 'rewards' in info:
            return np.array(info['rewards'])
        else:
            # Recompensa padrão: negativa do custo
            return -np.array([info.get('cost', 0)] * self.num_buildings)

    def _global_reward(self, observations, actions, info) -> np.ndarray:
        """Calcula recompensa baseada no estado global."""
        # Recompensa global compartilhada por todos os agentes
        global_reward = self._calculate_global_reward(observations, actions, info)

        # Distribuir igualmente entre todos os prédios
        return np.array([global_reward] * self.num_buildings)

    def _cooperative_reward(self, observations, actions, info) -> np.ndarray:
        """Calcula recompensa cooperativa (local + global + cooperação)."""
        # Componente local
        local_rewards = self._local_reward(observations, actions, info)

        # Componente global
        global_reward = self._calculate_global_reward(observations, actions, info)

        # Bônus de cooperação (simplificado por enquanto)
        cooperation_bonus = self._calculate_cooperation_bonus(actions)

        # Combinação ponderada
        weights = np.array([0.4, 0.4, 0.2])  # local, global, cooperation
        combined = (weights[0] * local_rewards +
                   weights[1] * global_reward +
                   weights[2] * cooperation_bonus)

        return combined

    def _calculate_global_reward(self, observations, actions, info) -> float:
        """Calcula recompensa global baseada no estado da rede."""
        # Por enquanto, usa métricas simples do CityLearn
        if hasattr(info, 'get'):
            # Penalizar custo total e pico de demanda
            total_cost = info.get('total_cost', 0)
            peak_demand = info.get('peak_demand', 0)

            return - (total_cost + peak_demand * 0.1)
        else:
            return 0.0

    def _calculate_cooperation_bonus(self, actions) -> np.ndarray:
        """Calcula bônus de cooperação entre agentes."""
        # Bônus simples: penalizar ações muito diferentes
        if len(actions) > 1:
            action_variance = np.var(actions)
            cooperation_bonus = -action_variance * 0.1  # Penalizar variância alta
        else:
            cooperation_bonus = 0.0

        return np.array([cooperation_bonus] * self.num_buildings)

    # Propriedades úteis

    @property
    def buildings(self):
        """Retorna lista de prédios do ambiente."""
        return self.citylearn_env.buildings

    @property
    def total_timesteps(self):
        """Retorna número total de passos no dataset."""
        return len(self.citylearn_env.buildings[0].observation_space.low)

    def get_building_info(self, building_id: int) -> Dict:
        """Retorna informações de um prédio específico."""
        if 0 <= building_id < self.num_buildings:
            building = self.citylearn_env.buildings[building_id]
            return {
                'id': building_id,
                'observation_space': building.observation_space,
                'action_space': building.action_space,
                'name': getattr(building, 'name', f'Building_{building_id}')
            }
        else:
            raise IndexError(f"Building ID {building_id} fora do range [0, {self.num_buildings-1}]")


def make_citylearn_vec_env(
    dataset_name: str = "citylearn_challenge_2022_phase_1",
    reward_function: str = "cooperative",
    communication: bool = True,
    normalize_obs: bool = True,
    max_episode_length: int = 8760,
    seed: Optional[int] = None,
    **kwargs
) -> CityLearnVecEnv:
    """
    Factory function para criar ambiente CityLearnVecEnv.

    Args:
        dataset_name: Nome do dataset CityLearn
        reward_function: Tipo de função de recompensa
        communication: Ativar comunicação entre agentes
        normalize_obs: Normalizar observações
        max_episode_length: Comprimento máximo do episódio
        seed: Seed para reprodutibilidade
        **kwargs: Argumentos adicionais

    Returns:
        CityLearnVecEnv: Ambiente configurado
    """
    return CityLearnVecEnv(
        dataset_name=dataset_name,
        reward_function=reward_function,
        communication=communication,
        normalize_obs=normalize_obs,
        max_episode_length=max_episode_length,
        seed=seed,
        **kwargs
    )