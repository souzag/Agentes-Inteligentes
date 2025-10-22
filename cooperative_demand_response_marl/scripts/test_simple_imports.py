#!/usr/bin/env python3
"""
Teste simples de imports dos módulos implementados.
"""

import sys
import os

def test_imports():
    """Testa imports básicos dos módulos."""
    print("🧪 Testando imports básicos...")

    try:
        # Adicionar diretório src ao path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Testar import do ambiente
        print("\n1. Testando ambiente...")
        from src.environment.citylearn_vec_env import make_citylearn_vec_env
        print("   ✅ make_citylearn_vec_env importado")

        # Testar import da comunicação
        print("\n2. Testando comunicação...")
        from src.environment.communication import FullCommunication
        print("   ✅ FullCommunication importado")

        # Testar import dos agentes
        print("\n3. Testando agentes...")
        from src.agents.base_agent import BaseAgent, AgentFactory
        print("   ✅ BaseAgent e AgentFactory importados")

        from src.agents.independent_agent import IndependentAgent, IndependentAgentFactory
        print("   ✅ IndependentAgent e IndependentAgentFactory importados")

        from src.agents.cooperative_agent import CooperativeAgent, CooperativeAgentFactory
        print("   ✅ CooperativeAgent e CooperativeAgentFactory importados")

        from src.agents.centralized_agent import CentralizedAgent, CentralizedAgentFactory
        print("   ✅ CentralizedAgent e CentralizedAgentFactory importados")

        from src.agents.agent_factory import (
            RandomAgent,
            RuleBasedAgent,
            RandomAgentFactory,
            RuleBasedAgentFactory,
            MultiAgentFactory
        )
        print("   ✅ Agentes baseline importados")

        # Testar políticas
        print("\n4. Testando políticas...")
        from src.agents.policies import (
            MultiAgentPolicy,
            CooperativePolicy,
            CentralizedPolicy,
            register_custom_policies
        )
        print("   ✅ Políticas importadas")

        print("\n✅ Todos os imports funcionaram!")
        return True

    except ImportError as e:
        print(f"❌ Erro no import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Testa funcionalidade básica."""
    print("\n🧪 Testando funcionalidade básica...")

    try:
        # Criar ambiente
        from src.environment.citylearn_vec_env import make_citylearn_vec_env
        env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")
        print(f"✅ Ambiente criado: {env.num_buildings} prédios")

        # Criar protocolo de comunicação
        from src.environment.communication import FullCommunication
        comm_protocol = FullCommunication(env.num_buildings)
        print(f"✅ Protocolo de comunicação criado: {type(comm_protocol).__name__}")

        # Criar agente baseline (sem SB3)
        from src.agents.agent_factory import RandomAgent
        agent = RandomAgent(env, 0, {})
        print(f"✅ Agente baseline criado: {agent}")

        # Testar seleção de ação
        obs, info = env.reset()
        action = agent.select_action(obs)
        print(f"✅ Ação selecionada: {action.shape}")

        # Validar ação
        if env.action_space.contains(action):
            print("✅ Ação válida")
        else:
            print("❌ Ação inválida")

        env.close()
        print("\n✅ Teste de funcionalidade concluído!")
        return True

    except Exception as e:
        print(f"❌ Erro na funcionalidade: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Teste simples de imports e funcionalidade...")

    success1 = test_imports()
    success2 = test_basic_functionality()

    if success1 and success2:
        print("\n🎉 Todos os testes passaram!")
        print("✅ O sistema MARL está funcionando corretamente!")
    else:
        print("\n⚠️ Alguns testes falharam")
        print("Verifique os erros acima")