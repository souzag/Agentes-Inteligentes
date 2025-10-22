#!/usr/bin/env python3
"""
Políticas customizadas para agentes MARL no sistema de demand response.

Este módulo implementa políticas customizadas para Stable Baselines3,
otimizadas para o ambiente CityLearn e cenários multi-agente cooperativos.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiAgentFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extrator de features customizado para múltiplos agentes.

    Este extrator processa observações de múltiplos agentes e opcionalmente
    incorpora informações de comunicação entre eles.
    """

    def __init__(self, observation_space, num_agents: int = 5, communication_dim: int = 0):
        """
        Inicializa o extrator de features.

        Args:
            observation_space: Espaço de observação
            num_agents: Número de agentes
            communication_dim: Dimensão do canal de comunicação
        """
        super().__init__(observation_space, features_dim=256)

        self.num_agents = num_agents
        self.communication_dim = communication_dim

        # Rede para processar observações individuais
        self.agent_network = nn.Sequential(
            nn.Linear(28, 64),  # 28 features por agente (CityLearn)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Rede para processar comunicação
        if communication_dim > 0:
            self.comm_network = nn.Sequential(
                nn.Linear(communication_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16)
            )

        # Rede para combinar features
        input_dim = 32 * num_agents + (16 if communication_dim > 0 else 0)
        self.combination_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 160),  # Ajustado para 160 para compatibilidade com action_space
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do extrator.

        Args:
            observations: Tensor de observações

        Returns:
            torch.Tensor: Features extraídas
        """
        batch_size = observations.shape[0]
        features_list = []

        # Processar cada agente
        for i in range(self.num_agents):
            agent_obs = observations[:, i*28:(i+1)*28]  # 28 features por agente
            agent_features = self.agent_network(agent_obs)
            features_list.append(agent_features)

        # Concatenar features de todos os agentes
        all_agent_features = torch.cat(features_list, dim=1)

        # Processar comunicação se disponível
        if self.communication_dim > 0:
            # Assumir que comunicação está nas últimas communication_dim features
            comm_features = observations[:, -self.communication_dim:]
            comm_processed = self.comm_network(comm_features)
            combined_features = torch.cat([all_agent_features, comm_processed], dim=1)
        else:
            combined_features = all_agent_features

        # Combinar todas as features
        final_features = self.combination_network(combined_features)

        return final_features


class MultiAgentPolicy(BasePolicy):
    """
    Política customizada para múltiplos agentes com comunicação opcional.

    Esta política implementa uma rede neural que processa observações de
    múltiplos agentes e opcionalmente incorpora comunicação entre eles.
    """

    def __init__(self, observation_space, action_space, num_agents: int = 5,
                 communication_dim: int = 0, shared_parameters: bool = True):
        """
        Inicializa a política multi-agente.

        Args:
            observation_space: Espaço de observação
            action_space: Espaço de ação
            num_agents: Número de agentes
            communication_dim: Dimensão do canal de comunicação
            shared_parameters: Se deve compartilhar parâmetros entre agentes
        """
        super().__init__(observation_space, action_space)

        self.num_agents = num_agents
        self.communication_dim = communication_dim
        self.shared_parameters = shared_parameters

        # Features extractor
        self.features_extractor = MultiAgentFeaturesExtractor(
            observation_space, num_agents, communication_dim
        )

        # Rede neural compartilhada
        self.shared_network = nn.Sequential(
            nn.Linear(160, 256),  # Ajustado para 160 entradas
            nn.ReLU(),
            nn.Linear(256, 160),
            nn.ReLU()
        )

        # Processamento de comunicação
        if communication_dim > 0:
            self.comm_network = nn.Sequential(
                nn.Linear(communication_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )

        # Cabeça da política
        input_dim = 160  # Compatível com o ambiente CityLearn (5 ações)
        self.action_network = nn.Linear(input_dim, action_space.shape[0])

        # Inicialização dos pesos
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Inicialização customizada dos pesos."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor, communication: Optional[torch.Tensor] = None):
        """
        Forward pass da política.

        Args:
            obs: Observações do ambiente
            communication: Tensor de comunicação (opcional)

        Returns:
            torch.Tensor: Ações
        """
        # Extrair features
        features = self.features_extractor(obs)

        # Processar features compartilhadas
        shared_features = self.shared_network(features)

        # Processar comunicação se disponível
        if communication is not None and self.communication_dim > 0:
            comm_features = self.comm_network(communication)
            combined = torch.cat([shared_features, comm_features], dim=-1)
        else:
            combined = shared_features

        # Gerar ação
        action = self.action_network(combined)
        return action

    def _predict(self, observation: torch.Tensor, deterministic: bool = True):
        """
        Prediz ação (método interno do SB3).

        Args:
            observation: Observação do ambiente
            deterministic: Se deve usar modo determinístico

        Returns:
            torch.Tensor: Ação predita
        """
        return self.forward(observation, None)

    def predict(self, observation: np.ndarray, communication: Optional[np.ndarray] = None,
                deterministic: bool = True):
        """
        Prediz ação baseada na observação.

        Args:
            observation: Observação do ambiente
            communication: Estado de comunicação (opcional)
            deterministic: Se deve usar modo determinístico

        Returns:
            Tuple: (ação, estado)
        """
        self.set_training_mode(False)

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        if communication is not None:
            comm_tensor = torch.tensor(communication, dtype=torch.float32).unsqueeze(0)
        else:
            comm_tensor = None

        with torch.no_grad():
            action = self.forward(obs_tensor, comm_tensor)

        action = action.squeeze(0).cpu().numpy()

        if deterministic:
            return action, None
        else:
            # Adicionar ruído para exploração
            noise = np.random.normal(0, 0.1, size=action.shape)
            return action + noise, None


class CooperativePolicy(MultiAgentPolicy):
    """
    Política otimizada para cooperação entre agentes.

    Esta política estende a MultiAgentPolicy com mecanismos específicos
    para incentivar a cooperação entre os agentes.
    """

    def __init__(self, observation_space, action_space, num_agents: int = 5,
                 communication_dim: int = 32, cooperation_strength: float = 0.1):
        """
        Inicializa a política cooperativa.

        Args:
            observation_space: Espaço de observação
            action_space: Espaço de ação
            num_agents: Número de agentes
            communication_dim: Dimensão do canal de comunicação
            cooperation_strength: Força da cooperação (0.0 a 1.0)
        """
        super().__init__(observation_space, action_space, num_agents, communication_dim)

        self.cooperation_strength = cooperation_strength

        # Camada adicional para processamento cooperativo
        self.cooperation_network = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()  # Limitar valores entre -1 e 1
        )

        # Mecanismo de atenção para comunicação
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

    def forward(self, obs: torch.Tensor, communication: Optional[torch.Tensor] = None):
        """Forward pass com mecanismos cooperativos."""
        # Processamento base
        features = self.features_extractor(obs)
        shared_features = self.shared_network(features)

        # Processar comunicação com atenção
        if communication is not None and self.communication_dim > 0:
            comm_features = self.comm_network(communication)

            # Aplicar atenção aos features de comunicação
            attention_output, _ = self.attention(comm_features.unsqueeze(0),
                                               comm_features.unsqueeze(0),
                                               comm_features.unsqueeze(0))
            attention_output = attention_output.squeeze(0)

            # Combinar com cooperação
            combined = torch.cat([shared_features, attention_output], dim=-1)
            coop_features = self.cooperation_network(combined)

            # Aplicar força de cooperação
            final_features = shared_features + self.cooperation_strength * coop_features
        else:
            final_features = shared_features

        # Gerar ação
        action = self.action_network(final_features)
        return action


class CentralizedPolicy(BasePolicy):
    """
    Política centralizada para controle global de todos os prédios.

    Esta política implementa uma rede neural que processa o estado global
    de todos os prédios e gera ações para todos simultaneamente.
    """

    def __init__(self, observation_space, action_space, num_buildings: int = 5):
        """
        Inicializa a política centralizada.

        Args:
            observation_space: Espaço de observação global
            action_space: Espaço de ação global
            num_buildings: Número de prédios controlados
        """
        super().__init__(observation_space, action_space)

        self.num_buildings = num_buildings

        # Rede neural para controle centralizado
        self.global_network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 160),  # Ajustado para 160 para compatibilidade
            nn.ReLU()
        )

        # Cabeças separadas para cada prédio
        self.building_heads = nn.ModuleList([
            nn.Linear(160, 1) for _ in range(num_buildings)
        ])

        # Inicialização
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Inicialização dos pesos."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """Forward pass para controle centralizado."""
        # Processar estado global
        global_features = self.global_network(obs)

        # Gerar ações para cada prédio
        actions = []
        for head in self.building_heads:
            action = head(global_features)
            actions.append(action)

        return torch.cat(actions, dim=-1)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True):
        """
        Prediz ação (método interno do SB3).

        Args:
            observation: Observação do ambiente
            deterministic: Se deve usar modo determinístico

        Returns:
            torch.Tensor: Ação predita
        """
        return self.forward(observation)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Prediz ações para todos os prédios."""
        self.set_training_mode(False)

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            actions = self.forward(obs_tensor)

        actions = actions.squeeze(0).cpu().numpy()

        if deterministic:
            return actions, None
        else:
            # Adicionar ruído para exploração
            noise = np.random.normal(0, 0.1, size=actions.shape)
            return actions + noise, None


class AttentionPolicy(BasePolicy):
    """
    Política baseada em mecanismos de atenção para comunicação.

    Esta política usa transformers/attention para processar comunicação
    entre agentes de forma mais sofisticada.
    """

    def __init__(self, observation_space, action_space, num_agents: int = 5,
                 communication_dim: int = 32, attention_heads: int = 4):
        """
        Inicializa a política com atenção.

        Args:
            observation_space: Espaço de observação
            action_space: Espaço de ação
            num_agents: Número de agentes
            communication_dim: Dimensão do canal de comunicação
            attention_heads: Número de cabeças de atenção
        """
        super().__init__(observation_space, action_space)

        self.num_agents = num_agents
        self.communication_dim = communication_dim
        self.attention_heads = attention_heads

        # Features extractor
        self.features_extractor = MultiAgentFeaturesExtractor(
            observation_space, num_agents, communication_dim
        )

        # Mecanismo de atenção para comunicação
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=attention_heads,
            batch_first=True
        )

        # Rede para processar saída da atenção
        self.attention_network = nn.Sequential(
            nn.Linear(160, 80),  # Ajustado para compatibilidade
            nn.ReLU(),
            nn.Linear(80, 40)
        )

        # Cabeça da política
        self.action_network = nn.Linear(160 + 40, action_space.shape[0])

        # Inicialização
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Inicialização dos pesos."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True):
        """
        Prediz ação (método interno do SB3).

        Args:
            observation: Observação do ambiente
            deterministic: Se deve usar modo determinístico

        Returns:
            torch.Tensor: Ação predita
        """
        return self.forward(observation, None)

    def forward(self, obs: torch.Tensor, communication: Optional[torch.Tensor] = None):
        """Forward pass com atenção."""
        # Extrair features
        features = self.features_extractor(obs)

        # Aplicar atenção se comunicação disponível
        if communication is not None and self.communication_dim > 0:
            # Usar features como query, key e value para atenção
            attention_output, _ = self.attention(features.unsqueeze(0),
                                                features.unsqueeze(0),
                                                features.unsqueeze(0))
            attention_output = attention_output.squeeze(0)

            # Processar saída da atenção
            attention_features = self.attention_network(attention_output)

            # Combinar features originais com atenção
            combined = torch.cat([features, attention_features], dim=-1)
        else:
            combined = features

        # Gerar ação
        action = self.action_network(combined)
        return action


# Registrar políticas no Stable Baselines3
def register_custom_policies():
    """Registra políticas customizadas no SB3."""
    try:
        # Nota: No SB3, políticas customizadas são registradas automaticamente
        # quando importadas. Não há necessidade de register_policy.
        print("✅ Políticas customizadas disponíveis:")
        print("   - MultiAgentPolicy")
        print("   - CooperativePolicy")
        print("   - CentralizedPolicy")
        print("   - AttentionPolicy")
        print("   (Políticas podem ser usadas diretamente com policy='MultiAgentPolicy')")

    except Exception as e:
        print(f"⚠️ Erro ao configurar políticas: {e}")


# Funções utilitárias para criação de políticas
def create_policy_from_config(policy_name: str, observation_space, action_space, config: Dict):
    """
    Cria política baseada na configuração.

    Args:
        policy_name: Nome da política
        observation_space: Espaço de observação
        action_space: Espaço de ação
        config: Configurações da política

    Returns:
        Política configurada
    """
    if policy_name == "MultiAgentPolicy":
        return MultiAgentPolicy(
            observation_space=observation_space,
            action_space=action_space,
            num_agents=config.get("num_agents", 5),
            communication_dim=config.get("communication_dim", 0),
            shared_parameters=config.get("shared_parameters", True)
        )
    elif policy_name == "CooperativePolicy":
        return CooperativePolicy(
            observation_space=observation_space,
            action_space=action_space,
            num_agents=config.get("num_agents", 5),
            communication_dim=config.get("communication_dim", 32),
            cooperation_strength=config.get("cooperation_strength", 0.1)
        )
    elif policy_name == "CentralizedPolicy":
        return CentralizedPolicy(
            observation_space=observation_space,
            action_space=action_space,
            num_buildings=config.get("num_buildings", 5)
        )
    elif policy_name == "AttentionPolicy":
        return AttentionPolicy(
            observation_space=observation_space,
            action_space=action_space,
            num_agents=config.get("num_agents", 5),
            communication_dim=config.get("communication_dim", 32),
            attention_heads=config.get("attention_heads", 4)
        )
    else:
        raise ValueError(f"Política não suportada: {policy_name}")


def get_policy_info(policy) -> Dict:
    """
    Retorna informações sobre uma política.

    Args:
        policy: Política a analisar

    Returns:
        Dict: Informações da política
    """
    info = {
        "policy_type": type(policy).__name__,
        "parameters": 0,
        "layers": []
    }

    # Contar parâmetros
    if hasattr(policy, 'parameters'):
        info["parameters"] = sum(p.numel() for p in policy.parameters())

    # Descrever arquitetura
    if hasattr(policy, 'shared_network'):
        info["layers"].append("shared_network")
    if hasattr(policy, 'comm_network'):
        info["layers"].append("comm_network")
    if hasattr(policy, 'attention'):
        info["layers"].append("attention")
    if hasattr(policy, 'action_network'):
        info["layers"].append("action_network")

    return info


# Testes das políticas
def test_policies():
    """Testa funcionalidades das políticas customizadas."""
    print("🧪 Testando políticas customizadas...")

    try:
        from src.environment import make_citylearn_vec_env

        # Criar ambiente
        env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")

        # Testar MultiAgentPolicy
        print("\n1. Testando MultiAgentPolicy...")
        policy = MultiAgentPolicy(
            env.observation_space,
            env.action_space,
            num_agents=5,
            communication_dim=32
        )

        obs, info = env.reset()
        action = policy.predict(obs)[0]

        assert env.action_space.contains(action), "Ação fora do espaço válido"
        print(f"   ✅ MultiAgentPolicy: {action.shape} ações geradas")

        # Testar CentralizedPolicy
        print("\n2. Testando CentralizedPolicy...")
        policy = CentralizedPolicy(
            env.observation_space,
            env.action_space,
            num_buildings=5
        )

        action = policy.predict(obs)[0]
        assert env.action_space.contains(action), "Ação fora do espaço válido"
        print(f"   ✅ CentralizedPolicy: {action.shape} ações geradas")

        # Testar CooperativePolicy
        print("\n3. Testando CooperativePolicy...")
        policy = CooperativePolicy(
            env.observation_space,
            env.action_space,
            num_agents=5,
            communication_dim=32,
            cooperation_strength=0.1
        )

        action = policy.predict(obs)[0]
        assert env.action_space.contains(action), "Ação fora do espaço válido"
        print(f"   ✅ CooperativePolicy: {action.shape} ações geradas")

        env.close()
        print("\n✅ Todas as políticas testadas com sucesso!")

        return True

    except Exception as e:
        print(f"❌ Erro nos testes: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Registrar políticas e executar testes
    register_custom_policies()
    test_policies()