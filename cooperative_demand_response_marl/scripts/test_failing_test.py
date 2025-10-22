#!/usr/bin/env python3
"""
Teste isolado para test_multi_agent_coordination.
"""

import sys
import os
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Configura o ambiente e importa dependências."""
    try:
        # Adicionar diretório src ao path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Importar componentes do ambiente
        from src.environment.citylearn_vec_env import make_citylearn_vec_env
        from src.environment.communication import FullCommunication

        # Importar componentes dos agentes
        from src.agents.base_agent import BaseAgent, AgentFactory
        from src.agents.independent_agent import IndependentAgent, IndependentAgentFactory
        from src.agents.cooperative_agent import CooperativeAgent, CooperativeAgentFactory
        from src.agents.centralized_agent import CentralizedAgent, CentralizedAgentFactory
        from src.agents.agent_factory import (
            RandomAgent,
            RuleBasedAgent,
            RandomAgentFactory,
            RuleBasedAgentFactory,
            MultiAgentFactory
        )

        print("✅ Ambiente de agentes configurado com sucesso")
        return {
            'make_citylearn_vec_env': make_citylearn_vec_env,
            'BaseAgent': BaseAgent,
            'IndependentAgent': IndependentAgent,
            'CooperativeAgent': CooperativeAgent,
            'CentralizedAgent': CentralizedAgent,
            'RandomAgent': RandomAgent,
            'RuleBasedAgent': RuleBasedAgent,
            'AgentFactory': AgentFactory,
            'IndependentAgentFactory': IndependentAgentFactory,
            'CooperativeAgentFactory': CooperativeAgentFactory,
            'CentralizedAgentFactory': CentralizedAgentFactory,
            'RandomAgentFactory': RandomAgentFactory,
            'RuleBasedAgentFactory': RuleBasedAgentFactory,
            'MultiAgentFactory': MultiAgentFactory,
            'FullCommunication': FullCommunication
        }

    except ImportError as e:
        print(f"❌ Erro ao importar dependências: {e}")
        print("Certifique-se de que todos os módulos estão implementados")
        return None

def test_multi_agent_coordination():
    """Testa coordenação entre múltiplos agentes."""
    print("\n=== Teste de Coordenação Multi-Agente ===")

    try:
        components = setup_environment()
        if components is None:
            return False

        env = components['make_citylearn_vec_env']("citylearn_challenge_2022_phase_1")

        # Criar sistema multi-agente cooperativo
        print("\n1. Criando sistema cooperativo...")
        comm_protocol = components['FullCommunication'](env.num_buildings)
        agents = components['CooperativeAgentFactory'].create_multi_agent_system(
            env, comm_protocol
        )
        print(f"   ✅ Sistema criado: {len(agents)} agentes cooperativos")

        # Testar coordenação
        print("\n2. Testando coordenação...")
        obs, info = env.reset()

        # Simular um passo com todos os agentes
        actions = []
        for i, agent in enumerate(agents):
            # Receber comunicação
            messages = agent.receive_messages()

            # Selecionar ação
            action = agent.select_action(obs, messages=messages)
            # Extrair ação específica para o agente (assumindo que cada agente controla um prédio)
            actions.append(action[i])

            # Enviar comunicação
            agent.send_communication(obs)

        print(f"   - Ações dos agentes: {actions}")

        # Executar ações (precisa ser lista de arrays 1D)
        obs, rewards, done, info = env.step(actions)
        print(f"   - Recompensas: {rewards}")
        print(f"   - Done: {done}")

        # Verificar se as ações foram coordenadas
        actions_array = np.array([a for a in actions])  # actions já são arrays 1D
        coordination = np.corrcoef(actions_array, actions_array)[0, 1] if len(actions) > 1 else 0.0
        print(f"   - Coordenação (correlação): {coordination:.3f}")

        env.close()
        return True

    except Exception as e:
        print(f"   ❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_agent_coordination()
    if success:
        print("\n✅ Teste passou!")
    else:
        print("\n❌ Teste falhou!")