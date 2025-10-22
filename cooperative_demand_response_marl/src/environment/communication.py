#!/usr/bin/env python3
"""
Sistema de comunicação entre agentes para o ambiente CityLearn.

Este módulo implementa diferentes protocolos de comunicação para permitir
a coordenação entre os prédios (agentes) no ambiente multi-agente.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum


class CommunicationType(Enum):
    """Tipos de comunicação disponíveis."""
    NONE = "none"           # Sem comunicação
    FULL = "full"           # Todos se comunicam com todos
    NEIGHBORHOOD = "neighborhood"  # Apenas vizinhos
    CENTRALIZED = "centralized"     # Coordenação centralizada
    HIERARCHICAL = "hierarchical"   # Hierarquia de comunicação


class Message:
    """
    Representa uma mensagem entre agentes.

    Attributes:
        sender_id (int): ID do agente remetente
        receiver_id (int): ID do agente destinatário
        content (Any): Conteúdo da mensagem
        timestamp (int): Timestamp da mensagem
        message_type (str): Tipo da mensagem
    """

    def __init__(self, sender_id: int, receiver_id: int, content: Any,
                 timestamp: int = 0, message_type: str = "state"):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.timestamp = timestamp
        self.message_type = message_type

    def __repr__(self):
        return f"Message(from={self.sender_id}, to={self.receiver_id}, type={self.message_type})"


class CommunicationProtocol(ABC):
    """
    Protocolo base para comunicação entre agentes.

    Esta classe define a interface para diferentes protocolos de comunicação
    que podem ser usados no ambiente multi-agente.
    """

    def __init__(self, num_agents: int, communication_type: CommunicationType = CommunicationType.FULL):
        """
        Inicializa o protocolo de comunicação.

        Args:
            num_agents: Número de agentes no ambiente
            communication_type: Tipo de comunicação
        """
        self.num_agents = num_agents
        self.communication_type = communication_type
        self.message_history = []
        self.max_history_size = 1000

    @abstractmethod
    def can_communicate(self, sender_id: int, receiver_id: int) -> bool:
        """Verifica se dois agentes podem se comunicar."""
        pass

    @abstractmethod
    def send_message(self, sender_id: int, receiver_id: int, content: Any,
                    timestamp: int = 0, message_type: str = "state") -> bool:
        """Envia mensagem entre agentes."""
        pass

    @abstractmethod
    def receive_messages(self, agent_id: int) -> List[Message]:
        """Recebe mensagens para um agente."""
        pass

    def get_communication_graph(self) -> np.ndarray:
        """Retorna matriz de conectividade de comunicação."""
        graph = np.zeros((self.num_agents, self.num_agents), dtype=bool)

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j and self.can_communicate(i, j):
                    graph[i, j] = True

        return graph


class FullCommunication(CommunicationProtocol):
    """
    Protocolo de comunicação completa - todos os agentes se comunicam com todos.

    Este é o protocolo mais simples e permite máxima coordenação, mas pode
    ser computacionalmente intensivo para muitos agentes.
    """

    def __init__(self, num_agents: int):
        super().__init__(num_agents, CommunicationType.FULL)

    def can_communicate(self, sender_id: int, receiver_id: int) -> bool:
        """Todos podem se comunicar com todos."""
        return sender_id != receiver_id

    def send_message(self, sender_id: int, receiver_id: int, content: Any,
                    timestamp: int = 0, message_type: str = "state") -> bool:
        """Envia mensagem para qualquer agente."""
        if not self.can_communicate(sender_id, receiver_id):
            return False

        message = Message(sender_id, receiver_id, content, timestamp, message_type)
        self.message_history.append(message)

        # Manter tamanho do histórico
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)

        return True

    def receive_messages(self, agent_id: int) -> List[Message]:
        """Recebe todas as mensagens para um agente."""
        agent_messages = [msg for msg in self.message_history
                         if msg.receiver_id == agent_id]

        # Remover mensagens processadas
        self.message_history = [msg for msg in self.message_history
                               if msg.receiver_id != agent_id]

        return agent_messages


class NeighborhoodCommunication(CommunicationProtocol):
    """
    Protocolo de comunicação por vizinhança - apenas agentes próximos se comunicam.

    Este protocolo é mais realista para redes elétricas reais onde apenas
    prédios próximos podem se coordenar diretamente.
    """

    def __init__(self, num_agents: int, max_distance: float = 100.0,
                 positions: Optional[np.ndarray] = None):
        """
        Inicializa comunicação por vizinhança.

        Args:
            num_agents: Número de agentes
            max_distance: Distância máxima para comunicação
            positions: Posições dos agentes (se None, usa distribuição aleatória)
        """
        super().__init__(num_agents, CommunicationType.NEIGHBORHOOD)
        self.max_distance = max_distance

        if positions is None:
            # Distribuir agentes aleatoriamente em uma área 200x200
            self.positions = np.random.rand(num_agents, 2) * 200
        else:
            self.positions = positions

        # Calcular matriz de distâncias
        self._calculate_distance_matrix()

    def _calculate_distance_matrix(self):
        """Calcula matriz de distâncias entre agentes."""
        self.distance_matrix = np.zeros((self.num_agents, self.num_agents))

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    self.distance_matrix[i, j] = dist

    def can_communicate(self, sender_id: int, receiver_id: int) -> bool:
        """Apenas vizinhos podem se comunicar."""
        if sender_id == receiver_id:
            return False

        distance = self.distance_matrix[sender_id, receiver_id]
        return distance <= self.max_distance

    def send_message(self, sender_id: int, receiver_id: int, content: Any,
                    timestamp: int = 0, message_type: str = "state") -> bool:
        """Envia mensagem apenas para vizinhos."""
        if not self.can_communicate(sender_id, receiver_id):
            return False

        message = Message(sender_id, receiver_id, content, timestamp, message_type)
        self.message_history.append(message)

        # Manter tamanho do histórico
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)

        return True

    def receive_messages(self, agent_id: int) -> List[Message]:
        """Recebe mensagens de vizinhos."""
        agent_messages = [msg for msg in self.message_history
                         if msg.receiver_id == agent_id]

        # Remover mensagens processadas
        self.message_history = [msg for msg in self.message_history
                               if msg.receiver_id != agent_id]

        return agent_messages

    def get_neighbors(self, agent_id: int) -> List[int]:
        """Retorna lista de vizinhos de um agente."""
        neighbors = []
        for i in range(self.num_agents):
            if i != agent_id and self.can_communicate(agent_id, i):
                neighbors.append(i)
        return neighbors


class CentralizedCommunication(CommunicationProtocol):
    """
    Protocolo de comunicação centralizada - um agente central coordena todos.

    Este protocolo usa um agente central (ou servidor) para coordenar
    as ações de todos os outros agentes.
    """

    def __init__(self, num_agents: int, coordinator_id: int = 0,
                 coordination_strategy: str = "consensus"):
        """
        Inicializa comunicação centralizada.

        Args:
            num_agents: Número de agentes
            coordinator_id: ID do agente coordenador
            coordination_strategy: Estratégia de coordenação
        """
        super().__init__(num_agents, CommunicationType.CENTRALIZED)
        self.coordinator_id = coordinator_id
        self.coordination_strategy = coordination_strategy

        # Estado global mantido pelo coordenador
        self.global_state = {}
        self.pending_actions = {}

    def can_communicate(self, sender_id: int, receiver_id: int) -> bool:
        """Todos se comunicam através do coordenador."""
        return True  # Todos podem enviar para o coordenador

    def send_message(self, sender_id: int, receiver_id: int, content: Any,
                    timestamp: int = 0, message_type: str = "state") -> bool:
        """Envia mensagem através do coordenador."""
        message = Message(sender_id, receiver_id, content, timestamp, message_type)
        self.message_history.append(message)

        # Atualizar estado global se for para o coordenador
        if receiver_id == self.coordinator_id:
            self._update_global_state(sender_id, content, message_type)

        # Manter tamanho do histórico
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)

        return True

    def receive_messages(self, agent_id: int) -> List[Message]:
        """Recebe mensagens (incluindo coordenação do central)."""
        agent_messages = [msg for msg in self.message_history
                         if msg.receiver_id == agent_id]

        # Adicionar coordenação do central se aplicável
        if self.coordination_strategy == "broadcast":
            coordination_msg = self._generate_coordination_message(agent_id)
            if coordination_msg:
                agent_messages.append(coordination_msg)

        # Remover mensagens processadas
        self.message_history = [msg for msg in self.message_history
                               if msg.receiver_id != agent_id]

        return agent_messages

    def _update_global_state(self, sender_id: int, content: Any, message_type: str):
        """Atualiza estado global mantido pelo coordenador."""
        if message_type == "state":
            self.global_state[sender_id] = content
        elif message_type == "action":
            self.pending_actions[sender_id] = content

    def _generate_coordination_message(self, agent_id: int) -> Optional[Message]:
        """Gera mensagem de coordenação do central."""
        if len(self.global_state) == self.num_agents:
            # Todos os estados recebidos, gerar coordenação
            coordination_info = self._coordinate_actions()

            return Message(
                sender_id=self.coordinator_id,
                receiver_id=agent_id,
                content=coordination_info,
                message_type="coordination"
            )
        return None

    def _coordinate_actions(self) -> Dict:
        """Coordena ações baseado na estratégia."""
        if self.coordination_strategy == "consensus":
            return self._consensus_coordination()
        elif self.coordination_strategy == "auction":
            return self._auction_coordination()
        else:
            return {}

    def _consensus_coordination(self) -> Dict:
        """Coordenação por consenso."""
        # Implementação simplificada: média das ações
        if self.pending_actions:
            avg_action = np.mean(list(self.pending_actions.values()))
            return {"consensus_action": avg_action}
        return {}

    def _auction_coordination(self) -> Dict:
        """Coordenação por leilão."""
        # Implementação simplificada: priorizar ações mais eficientes
        if self.pending_actions:
            best_action = min(self.pending_actions.items(),
                            key=lambda x: abs(x[1]))  # Menor magnitude
            return {"auction_winner": best_action[0], "winning_action": best_action[1]}
        return {}


class HierarchicalCommunication(CommunicationProtocol):
    """
    Protocolo de comunicação hierárquica - agentes organizados em hierarquia.

    Este protocolo organiza os agentes em uma estrutura hierárquica onde
    agentes de nível superior coordenam agentes de nível inferior.
    """

    def __init__(self, num_agents: int, hierarchy_levels: int = 2,
                 cluster_size: int = 3):
        """
        Inicializa comunicação hierárquica.

        Args:
            num_agents: Número de agentes
            hierarchy_levels: Número de níveis na hierarquia
            cluster_size: Tamanho de cada cluster
        """
        super().__init__(num_agents, CommunicationType.HIERARCHICAL)
        self.hierarchy_levels = hierarchy_levels
        self.cluster_size = cluster_size

        # Organizar agentes em hierarquia
        self.hierarchy = self._build_hierarchy()

    def _build_hierarchy(self) -> Dict[int, Dict]:
        """Constrói estrutura hierárquica."""
        hierarchy = {}

        # Dividir agentes em clusters
        clusters = []
        for i in range(0, self.num_agents, self.cluster_size):
            cluster = list(range(i, min(i + self.cluster_size, self.num_agents)))
            clusters.append(cluster)

        # Construir hierarquia
        for level in range(self.hierarchy_levels):
            level_clusters = {}

            if level == 0:
                # Nível mais baixo: agentes individuais
                for i in range(self.num_agents):
                    level_clusters[i] = {"parent": None, "children": []}
            else:
                # Níveis superiores: clusters
                for cluster_id, cluster in enumerate(clusters):
                    level_clusters[cluster_id] = {"parent": None, "children": cluster}

                    # Definir pais dos agentes
                    for agent_id in cluster:
                        hierarchy[level-1][agent_id]["parent"] = cluster_id

            hierarchy[level] = level_clusters

        self.hierarchy = hierarchy
        return hierarchy

    def can_communicate(self, sender_id: int, receiver_id: int) -> bool:
        """Apenas agentes no mesmo cluster ou níveis adjacentes podem se comunicar."""
        if sender_id == receiver_id:
            return False

        # Encontrar clusters dos agentes
        sender_cluster = self._find_agent_cluster(sender_id)
        receiver_cluster = self._find_agent_cluster(receiver_id)

        # Mesma hierarquia ou níveis adjacentes
        return (sender_cluster == receiver_cluster or
                abs(self._get_agent_level(sender_id) - self._get_agent_level(receiver_id)) <= 1)

    def _find_agent_cluster(self, agent_id: int) -> Optional[int]:
        """Encontra cluster de um agente."""
        for cluster_id, cluster in enumerate(self.hierarchy[0].values()):
            if agent_id in cluster.get("children", []):
                return cluster_id
        return None

    def _get_agent_level(self, agent_id: int) -> int:
        """Retorna nível hierárquico de um agente."""
        for level, level_clusters in self.hierarchy.items():
            for cluster in level_clusters.values():
                if agent_id in cluster.get("children", []):
                    return level
        return -1

    def send_message(self, sender_id: int, receiver_id: int, content: Any,
                    timestamp: int = 0, message_type: str = "state") -> bool:
        """Envia mensagem respeitando hierarquia."""
        if not self.can_communicate(sender_id, receiver_id):
            return False

        message = Message(sender_id, receiver_id, content, timestamp, message_type)
        self.message_history.append(message)

        # Manter tamanho do histórico
        if len(self.message_history) > self.max_history_size:
            self.message_history.pop(0)

        return True

    def receive_messages(self, agent_id: int) -> List[Message]:
        """Recebe mensagens respeitando hierarquia."""
        agent_messages = [msg for msg in self.message_history
                         if msg.receiver_id == agent_id]

        # Remover mensagens processadas
        self.message_history = [msg for msg in self.message_history
                               if msg.receiver_id != agent_id]

        return agent_messages


# Factory function para criar protocolos de comunicação
def create_communication_protocol(protocol_type: str, num_agents: int, **kwargs) -> CommunicationProtocol:
    """
    Factory function para criar protocolos de comunicação.

    Args:
        protocol_type: Tipo de protocolo ("full", "neighborhood", "centralized", "hierarchical")
        num_agents: Número de agentes
        **kwargs: Argumentos específicos do protocolo

    Returns:
        CommunicationProtocol: Protocolo configurado
    """
    if protocol_type == "full":
        return FullCommunication(num_agents)
    elif protocol_type == "neighborhood":
        return NeighborhoodCommunication(num_agents, **kwargs)
    elif protocol_type == "centralized":
        return CentralizedCommunication(num_agents, **kwargs)
    elif protocol_type == "hierarchical":
        return HierarchicalCommunication(num_agents, **kwargs)
    else:
        raise ValueError(f"Tipo de protocolo inválido: {protocol_type}")


# Funções utilitárias para análise de comunicação
def analyze_communication_network(protocol: CommunicationProtocol) -> Dict:
    """
    Analisa características da rede de comunicação.

    Args:
        protocol: Protocolo de comunicação

    Returns:
        Dict: Métricas da rede
    """
    graph = protocol.get_communication_graph()

    # Calcular métricas
    num_connections = np.sum(graph)
    avg_connections = num_connections / protocol.num_agents
    max_connections = np.max(np.sum(graph, axis=1))
    min_connections = np.min(np.sum(graph, axis=1))

    # Conectividade
    is_connected = _is_network_connected(graph)

    return {
        "num_agents": protocol.num_agents,
        "communication_type": protocol.communication_type.value,
        "total_connections": int(num_connections),
        "avg_connections_per_agent": float(avg_connections),
        "max_connections": int(max_connections),
        "min_connections": int(min_connections),
        "is_connected": is_connected,
        "connectivity_ratio": float(num_connections) / (protocol.num_agents * (protocol.num_agents - 1))
    }


def _is_network_connected(graph: np.ndarray) -> bool:
    """Verifica se a rede de comunicação é conectada."""
    # Implementação simplificada usando BFS
    if graph.size == 0:
        return True

    visited = np.zeros(graph.shape[0], dtype=bool)
    queue = [0]  # Começar do primeiro agente
    visited[0] = True

    while queue:
        current = queue.pop(0)

        # Encontrar vizinhos não visitados
        neighbors = np.where(graph[current] & ~visited)[0]

        for neighbor in neighbors:
            visited[neighbor] = True
            queue.append(neighbor)

    return np.all(visited)


def optimize_communication_topology(num_agents: int, target_connectivity: float = 0.5) -> Dict:
    """
    Otimiza topologia de comunicação para um número alvo de conexões.

    Args:
        num_agents: Número de agentes
        target_connectivity: Razão de conectividade desejada (0.0 a 1.0)

    Returns:
        Dict: Configuração otimizada
    """
    max_possible_connections = num_agents * (num_agents - 1)

    if target_connectivity <= 0.3:
        # Topologia esparsa: vizinhança local
        return {
            "protocol_type": "neighborhood",
            "max_distance": 50.0,
            "positions": np.random.rand(num_agents, 2) * 100
        }
    elif target_connectivity <= 0.7:
        # Topologia moderada: clusters
        return {
            "protocol_type": "hierarchical",
            "hierarchy_levels": 2,
            "cluster_size": max(2, num_agents // 3)
        }
    else:
        # Topologia densa: comunicação completa
        return {
            "protocol_type": "full"
        }