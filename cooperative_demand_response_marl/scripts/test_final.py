#!/usr/bin/env python3
"""
Teste final completo do sistema MARL.
"""

import sys
import os

def test_complete_system():
    """Testa o sistema completo de forma rápida."""
    print("🚀 Teste Final do Sistema MARL Completo")
    print("=" * 50)

    try:
        # Adicionar diretório src ao path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # 1. Testar ambiente
        print("\n1. 🏢 Testando Ambiente...")
        from src.environment.citylearn_vec_env import make_citylearn_vec_env
        env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")
        print(f"   ✅ Ambiente criado: {env.num_buildings} prédios")

        # 2. Testar comunicação
        print("\n2. 📡 Testando Comunicação...")
        from src.environment.communication import FullCommunication
        comm_protocol = FullCommunication(env.num_buildings)
        print(f"   ✅ Protocolo criado: {type(comm_protocol).__name__}")

        # 3. Testar agentes baseline
        print("\n3. 🤖 Testando Agentes Baseline...")
        from src.agents.agent_factory import RandomAgent, RuleBasedAgent

        random_agent = RandomAgent(env, 0, {})
        rule_agent = RuleBasedAgent(env, 1, {})

        obs, info = env.reset()
        action1 = random_agent.select_action(obs)
        action2 = rule_agent.select_action(obs)

        print(f"   ✅ RandomAgent: ação {action1.shape}")
        print(f"   ✅ RuleBasedAgent: ação {action2.shape}")

        # 4. Testar agentes SB3
        print("\n4. 🧠 Testando Agentes SB3...")
        from src.agents.independent_agent import IndependentAgent
        from src.agents.cooperative_agent import CooperativeAgent
        from src.agents.centralized_agent import CentralizedAgent

        # IndependentAgent
        ind_agent = IndependentAgent(env, 2, {})
        action = ind_agent.select_action(obs)
        print(f"   ✅ IndependentAgent: ação {action.shape}")

        # CooperativeAgent
        coop_agent = CooperativeAgent(env, 3, {}, comm_protocol)
        action = coop_agent.select_action(obs)
        print(f"   ✅ CooperativeAgent: ação {action.shape}")

        # CentralizedAgent
        central_agent = CentralizedAgent(env, 4, {})
        action = central_agent.select_action(obs)
        print(f"   ✅ CentralizedAgent: ação {action.shape}")

        # 5. Testar políticas customizadas
        print("\n5. 🎯 Testando Políticas Customizadas...")
        from src.agents.policies import MultiAgentPolicy, CentralizedPolicy

        # MultiAgentPolicy
        policy = MultiAgentPolicy(env.observation_space, env.action_space, num_agents=5)
        action = policy.predict(obs)[0]
        print(f"   ✅ MultiAgentPolicy: ação {action.shape}")

        # CentralizedPolicy
        policy = CentralizedPolicy(env.observation_space, env.action_space, num_buildings=5)
        action = policy.predict(obs)[0]
        print(f"   ✅ CentralizedPolicy: ação {action.shape}")

        # 6. Testar factory functions
        print("\n6. 🏭 Testando Factory Functions...")
        from src.agents.base_agent import AgentFactory

        config = {"agent_type": "independent"}
        agents = AgentFactory.create_multi_agent_system(env, config)
        print(f"   ✅ Sistema criado: {len(agents)} agentes")

        env.close()

        print("\n" + "=" * 50)
        print("🎉 TESTE FINAL CONCLUÍDO COM SUCESSO!")
        print("✅ Todos os componentes estão funcionando!")
        print("✅ Sistema MARL completamente operacional!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n❌ ERRO no teste final: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()

    if success:
        print("\n🏆 SISTEMA MARL TOTALMENTE FUNCIONAL!")
        print("🚀 Pronto para treinamento de modelos cooperativos!")
        sys.exit(0)
    else:
        print("\n⚠️ Alguns componentes ainda precisam ajustes")
        sys.exit(1)