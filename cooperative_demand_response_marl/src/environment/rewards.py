#!/usr/bin/env python3
"""
Sistema de recompensas cooperativas para o ambiente CityLearn.

Este módulo implementa diferentes funções de recompensa para incentivar
comportamentos cooperativos entre os agentes (prédios) no ambiente CityLearn.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """
    Classe base abstrata para funções de recompensa.

    Todas as funções de recompensa devem herdar desta classe e implementar
    o método __call__ para calcular as recompensas.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Inicializa a função de recompensa.

        Args:
            weights: Pesos para diferentes componentes da recompensa
        """
        self.weights = weights or {
            "comfort": 0.25,
            "cost": 0.25,
            "peak": 0.25,
            "cooperation": 0.25
        }

    @abstractmethod
    def __call__(self, observations: Any, actions: List, info: Dict) -> np.ndarray:
        """
        Calcula recompensas para todos os prédios.

        Args:
            observations: Observações do ambiente
            actions: Ações executadas
            info: Informações adicionais do ambiente

        Returns:
            np.ndarray: Recompensas para cada prédio
        """
        pass


class LocalReward(RewardFunction):
    """
    Função de recompensa baseada apenas no estado individual de cada prédio.

    Componentes:
    - Conforto térmico: penalidade por desvio de temperatura
    - Custos energéticos: custo de eletricidade
    - Eficiência: uso eficiente de recursos
    """

    def __call__(self, observations: Any, actions: List, info: Dict) -> np.ndarray:
        """
        Calcula recompensa local para cada prédio.

        Args:
            observations: Observações do ambiente
            actions: Ações executadas
            info: Informações adicionais

        Returns:
            np.ndarray: Recompensas locais para cada prédio
        """
        rewards = []

        for i, (obs, action) in enumerate(zip(observations, actions)):
            # Componente de conforto
            comfort_penalty = self._comfort_penalty(obs)

            # Componente de custo
            cost_penalty = self._cost_penalty(obs, action, info)

            # Componente de eficiência
            efficiency_bonus = self._efficiency_bonus(obs, action)

            # Recompensa total para o prédio
            reward = -comfort_penalty - cost_penalty + efficiency_bonus
            rewards.append(reward)

        return np.array(rewards)

    def _comfort_penalty(self, obs: np.ndarray) -> float:
        """Calcula penalidade por desconforto térmico."""
        # Assumindo que as features de temperatura estão em posições específicas
        # Baseado na análise do CityLearn: features 3-6 são temperaturas
        if len(obs) >= 7:
            indoor_temp = obs[3:7].mean()  # Temperatura interna média
            comfort_range = (20.0, 26.0)  # Faixa de conforto

            if indoor_temp < comfort_range[0]:
                penalty = (comfort_range[0] - indoor_temp) ** 2
            elif indoor_temp > comfort_range[1]:
                penalty = (indoor_temp - comfort_range[1]) ** 2
            else:
                penalty = 0.0

            return penalty * 0.1  # Escala da penalidade
        else:
            return 0.0

    def _cost_penalty(self, obs: np.ndarray, action: float, info: Dict) -> float:
        """Calcula penalidade por custos energéticos."""
        # Usar informação de custo do ambiente se disponível
        building_cost = info.get(f'building_{len(obs)}_cost', 0)

        # Penalizar ações extremas (custos operacionais)
        action_penalty = abs(action) * 0.01

        return building_cost + action_penalty

    def _efficiency_bonus(self, obs: np.ndarray, action: float) -> float:
        """Calcula bônus por eficiência energética."""
        # Bônus por uso de energia renovável (se disponível)
        if len(obs) >= 10:
            solar_generation = obs[8:10].sum()  # Geração solar
            efficiency_bonus = solar_generation * 0.001
        else:
            efficiency_bonus = 0.0

        return efficiency_bonus


class GlobalReward(RewardFunction):
    """
    Função de recompensa baseada no estado global da rede elétrica.

    Componentes:
    - Balanceamento: redução de picos de demanda
    - Emissões: redução de carbono
    - Estabilidade: suavização da curva de carga
    """

    def __call__(self, observations: Any, actions: List, info: Dict) -> np.ndarray:
        """
        Calcula recompensa global compartilhada por todos os prédios.

        Args:
            observations: Observações do ambiente
            actions: Ações executadas
            info: Informações adicionais

        Returns:
            np.ndarray: Recompensa global para cada prédio
        """
        # Calcular recompensa global
        global_reward = self._calculate_global_reward(observations, actions, info)

        # Distribuir igualmente entre todos os prédios
        num_buildings = len(observations)
        return np.array([global_reward] * num_buildings)

    def _calculate_global_reward(self, observations: Any, actions: List, info: Dict) -> float:
        """Calcula recompensa baseada no estado global."""
        # Componente de pico de demanda
        peak_penalty = self._peak_demand_penalty(observations, info)

        # Componente de emissões de carbono
        carbon_penalty = self._carbon_emissions_penalty(observations, info)

        # Componente de balanceamento
        balancing_bonus = self._load_balancing_bonus(observations, actions)

        # Componente de estabilidade
        stability_bonus = self._grid_stability_bonus(actions)

        return -peak_penalty - carbon_penalty + balancing_bonus + stability_bonus

    def _peak_demand_penalty(self, observations: Any, info: Dict) -> float:
        """Calcula penalidade por pico de demanda."""
        # Usar informação de pico do ambiente
        peak_demand = info.get('peak_demand', 0)
        return peak_demand * 0.1

    def _carbon_emissions_penalty(self, observations: Any, info: Dict) -> float:
        """Calcula penalidade por emissões de carbono."""
        # Usar informação de emissões do ambiente
        carbon_emissions = info.get('carbon_emissions', 0)
        return carbon_emissions * 0.05

    def _load_balancing_bonus(self, observations: Any, actions: List) -> float:
        """Calcula bônus por balanceamento da carga."""
        # Bônus por distribuição uniforme de ações
        if len(actions) > 1:
            action_std = np.std(actions)
            # Bônus por baixa variabilidade (ações coordenadas)
            balancing_bonus = -action_std * 0.1
        else:
            balancing_bonus = 0.0

        return balancing_bonus

    def _grid_stability_bonus(self, actions: List) -> float:
        """Calcula bônus por estabilidade da rede."""
        # Bônus por ações suaves (sem mudanças bruscas)
        if len(actions) > 1:
            action_smoothness = self._calculate_action_smoothness(actions)
            stability_bonus = action_smoothness * 0.05
        else:
            stability_bonus = 0.0

        return stability_bonus

    def _calculate_action_smoothness(self, actions: List) -> float:
        """Calcula suavidade das ações (menor é melhor)."""
        actions = np.array(actions)
        if len(actions) < 2:
            return 0.0

        # Calcular diferenças entre ações consecutivas
        differences = np.abs(np.diff(actions))
        return 1.0 / (1.0 + np.mean(differences))


class CooperativeReward(RewardFunction):
    """
    Função de recompensa cooperativa que combina componentes locais e globais.

    Esta é a função de recompensa mais avançada, incentivando tanto o
    desempenho individual quanto a cooperação entre os prédios.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Inicializa função de recompensa cooperativa."""
        super().__init__(weights)

        # Componentes de recompensa
        self.local_reward_fn = LocalReward()
        self.global_reward_fn = GlobalReward()

    def __call__(self, observations: Any, actions: List, info: Dict) -> np.ndarray:
        """
        Calcula recompensa cooperativa.

        Args:
            observations: Observações do ambiente
            actions: Ações executadas
            info: Informações adicionais

        Returns:
            np.ndarray: Recompensas cooperativas para cada prédio
        """
        # Componente local
        local_rewards = self.local_reward_fn(observations, actions, info)

        # Componente global
        global_reward = self.global_reward_fn(observations, actions, info)[0]

        # Componente de cooperação
        cooperation_bonus = self._cooperation_bonus(observations, actions, info)

        # Combinação ponderada
        combined_reward = (
            self.weights["comfort"] * local_rewards +
            self.weights["cost"] * (-np.array([info.get('total_cost', 0)] * len(local_rewards))) +
            self.weights["peak"] * global_reward +
            self.weights["cooperation"] * cooperation_bonus
        )

        return combined_reward

    def _cooperation_bonus(self, observations: Any, actions: List, info: Dict) -> np.ndarray:
        """Calcula bônus de cooperação entre prédios."""
        # Bônus por coordenação de ações
        coordination_bonus = self._action_coordination_bonus(actions)

        # Bônus por compartilhamento de recursos
        resource_sharing_bonus = self._resource_sharing_bonus(observations)

        # Bônus por objetivos comuns
        common_goal_bonus = self._common_goal_bonus(observations, info)

        total_bonus = coordination_bonus + resource_sharing_bonus + common_goal_bonus

        return np.array([total_bonus] * len(actions))

    def _action_coordination_bonus(self, actions: List) -> float:
        """Calcula bônus por coordenação de ações."""
        actions = np.array(actions)

        if len(actions) < 2:
            return 0.0

        # Medir similaridade das ações (ações similares = mais coordenadas)
        action_correlation = np.corrcoef(actions, actions)[0, 1] if len(actions) > 1 else 0.0

        # Bônus por correlação positiva (ações coordenadas)
        return max(0.0, action_correlation) * 0.1

    def _resource_sharing_bonus(self, observations: Any) -> float:
        """Calcula bônus por compartilhamento de recursos."""
        # Bônus por uso eficiente de armazenamento compartilhado
        storage_bonus = 0.0

        for obs in observations:
            if len(obs) >= 15:  # Features de armazenamento
                storage_levels = obs[10:15]  # Estado dos armazenamentos
                # Bônus por níveis de armazenamento balanceados
                storage_balance = 1.0 - np.std(storage_levels)
                storage_bonus += storage_balance * 0.01

        return storage_bonus / len(observations)

    def _common_goal_bonus(self, observations: Any, info: Dict) -> float:
        """Calcula bônus por objetivos comuns."""
        # Bônus por redução coletiva de custos
        total_cost_reduction = info.get('cost_reduction', 0)

        # Bônus por melhoria da eficiência global
        efficiency_improvement = info.get('efficiency_improvement', 0)

        return (total_cost_reduction + efficiency_improvement) * 0.1


class AdaptiveReward(RewardFunction):
    """
    Função de recompensa adaptativa que ajusta pesos dinamicamente.

    Esta função ajusta os pesos das componentes de recompensa baseado
    no desempenho histórico e estado atual do ambiente.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Inicializa função de recompensa adaptativa."""
        super().__init__(weights)

        # Histórico para adaptação
        self.reward_history = []
        self.performance_history = []

        # Parâmetros de adaptação
        self.adaptation_rate = 0.1
        self.min_weight = 0.1
        self.max_weight = 0.5

    def __call__(self, observations: Any, actions: List, info: Dict) -> np.ndarray:
        """Calcula recompensa adaptativa."""
        # Calcular recompensa usando pesos atuais
        cooperative_fn = CooperativeReward(self.weights)
        rewards = cooperative_fn(observations, actions, info)

        # Atualizar pesos baseado no desempenho
        self._update_weights(observations, actions, info, rewards)

        return rewards

    def _update_weights(self, observations: Any, actions: List, info: Dict, rewards: np.ndarray):
        """Atualiza pesos baseado no desempenho."""
        # Calcular métricas de desempenho
        performance_metrics = self._calculate_performance_metrics(observations, actions, info)

        # Armazenar histórico
        self.reward_history.append(np.mean(rewards))
        self.performance_history.append(performance_metrics)

        # Manter apenas últimos 100 valores
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
            self.performance_history.pop(0)

        # Ajustar pesos se necessário
        if len(self.reward_history) > 10:
            self._adapt_weights(performance_metrics)

    def _calculate_performance_metrics(self, observations: Any, actions: List, info: Dict) -> Dict:
        """Calcula métricas de desempenho."""
        return {
            'comfort': self._average_comfort(observations),
            'cost': info.get('total_cost', 0),
            'peak': info.get('peak_demand', 0),
            'cooperation': self._cooperation_metric(actions)
        }

    def _average_comfort(self, observations: Any) -> float:
        """Calcula conforto médio."""
        comfort_scores = []
        for obs in observations:
            comfort_penalty = LocalReward()._comfort_penalty(obs)
            comfort_score = 1.0 / (1.0 + comfort_penalty)
            comfort_scores.append(comfort_score)

        return np.mean(comfort_scores)

    def _cooperation_metric(self, actions: List) -> float:
        """Calcula métrica de cooperação."""
        if len(actions) < 2:
            return 1.0

        # Correlação das ações como métrica de cooperação
        correlation = np.corrcoef(actions, actions)[0, 1] if len(actions) > 1 else 0.0
        return max(0.0, correlation)

    def _adapt_weights(self, current_performance: Dict):
        """Adapta pesos baseado no desempenho atual."""
        # Identificar componente com pior desempenho
        worst_component = min(current_performance.keys(),
                            key=lambda k: current_performance[k])

        # Aumentar peso do componente com pior desempenho
        if worst_component in self.weights:
            # Aumentar peso do pior componente
            increase = self.adaptation_rate * (self.max_weight - self.weights[worst_component])
            self.weights[worst_component] += increase

            # Diminuir pesos dos outros componentes proporcionalmente
            other_components = [k for k in self.weights.keys() if k != worst_component]
            total_decrease = increase / len(other_components)

            for component in other_components:
                decrease = min(total_decrease, self.weights[component] - self.min_weight)
                self.weights[component] -= decrease
                self.weights[worst_component] += decrease / len(other_components)

        # Normalizar pesos
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight


# Factory function para criar funções de recompensa
def create_reward_function(reward_type: str, **kwargs) -> RewardFunction:
    """
    Factory function para criar funções de recompensa.

    Args:
        reward_type: Tipo de função de recompensa
        **kwargs: Argumentos adicionais

    Returns:
        RewardFunction: Função de recompensa configurada
    """
    if reward_type == "local":
        return LocalReward(**kwargs)
    elif reward_type == "global":
        return GlobalReward(**kwargs)
    elif reward_type == "cooperative":
        return CooperativeReward(**kwargs)
    elif reward_type == "adaptive":
        return AdaptiveReward(**kwargs)
    else:
        raise ValueError(f"Tipo de recompensa inválido: {reward_type}")


# Funções utilitárias para análise de recompensas
def analyze_reward_components(reward_function: RewardFunction,
                            observations: Any, actions: List, info: Dict) -> Dict:
    """
    Analisa componentes individuais de uma função de recompensa.

    Args:
        reward_function: Função de recompensa a analisar
        observations: Observações do ambiente
        actions: Ações executadas
        info: Informações adicionais

    Returns:
        Dict: Componentes da recompensa
    """
    if isinstance(reward_function, CooperativeReward):
        local_rewards = reward_function.local_reward_fn(observations, actions, info)
        global_reward = reward_function.global_reward_fn(observations, actions, info)[0]
        cooperation_bonus = reward_function._cooperation_bonus(observations, actions, info)

        return {
            "local": local_rewards,
            "global": np.array([global_reward] * len(local_rewards)),
            "cooperation": cooperation_bonus,
            "total": reward_function(observations, actions, info)
        }
    else:
        return {
            "total": reward_function(observations, actions, info)
        }