#!/usr/bin/env python3
"""
Teste específico para agentes com SB3.
"""

import sys
import os

def test_sb3_agents():
    """Testa agentes com SB3."""
    print("🧪 Testando agentes com SB3...")

    try:
        # Adicionar diretório src ao path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Criar ambiente
        from src.environment.citylearn_vec_env import make_citylearn_vec_env
        env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")
        print(f"✅ Ambiente criado: {env.num_buildings} prédios")

        # Testar IndependentAgent
        print("\n1. Testando IndependentAgent...")
        from src.agents.independent_agent import IndependentAgent

        config = {
            "learning_rate": 3e-4,
            "activation_fn": "Tanh",
            "policy_kwargs": {
                "net_arch": [64, 64],
                "activation_fn": "tanh"
            }
        }

        agent = IndependentAgent(env, 0, config)
        print(f"   ✅ IndependentAgent criado: {agent}")

        # Testar seleção de ação
        obs, info = env.reset()
        action = agent.select_action(obs)
        print(f"   ✅ Ação selecionada: {action.shape}")

        # Validar ação
        if env.action_space.contains(action):
            print("   ✅ Ação válida")
        else:
            print("   ❌ Ação inválida")
            return False

        # Testar CooperativeAgent
        print("\n2. Testando CooperativeAgent...")
        from src.agents.cooperative_agent import CooperativeAgent
        from src.environment.communication import FullCommunication

        comm_protocol = FullCommunication(env.num_buildings)
        config = {
            "learning_rate": 3e-4,
            "activation_fn": "ReLU",
            "cooperation_strength": 0.1,
            "policy_kwargs": {
                "net_arch": [256, 256, 128],
                "activation_fn": "relu"
            }
        }

        agent = CooperativeAgent(env, 1, config, comm_protocol)
        print(f"   ✅ CooperativeAgent criado: {agent}")

        # Testar seleção de ação
        action = agent.select_action(obs)
        print(f"   ✅ Ação selecionada: {action.shape}")

        # Validar ação
        if env.action_space.contains(action):
            print("   ✅ Ação válida")
        else:
            print("   ❌ Ação inválida")
            return False

        # Testar CentralizedAgent
        print("\n3. Testando CentralizedAgent...")
        from src.agents.centralized_agent import CentralizedAgent

        config = {
            "learning_rate": 3e-4,
            "activation_fn": "ReLU",
            "coordination_strategy": "global",
            "policy_kwargs": {
                "net_arch": [512, 256, 128],
                "activation_fn": "relu"
            }
        }

        agent = CentralizedAgent(env, 2, config)
        print(f"   ✅ CentralizedAgent criado: {agent}")

        # Testar seleção de ação
        action = agent.select_action(obs)
        print(f"   ✅ Ação selecionada: {action.shape}")

        # Validar ação
        if env.action_space.contains(action):
            print("   ✅ Ação válida")
        else:
            print("   ❌ Ação inválida")
            return False

        env.close()
        print("\n✅ Todos os agentes SB3 testados com sucesso!")
        return True

    except Exception as e:
        print(f"❌ Erro nos testes SB3: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_policies():
    """Testa políticas customizadas."""
    print("\n🧪 Testando políticas customizadas...")

    try:
        # Adicionar diretório src ao path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Criar ambiente
        from src.environment.citylearn_vec_env import make_citylearn_vec_env
        env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")

        # Testar MultiAgentPolicy
        print("\n1. Testando MultiAgentPolicy...")
        from src.agents.policies import MultiAgentPolicy

        policy = MultiAgentPolicy(
            env.observation_space,
            env.action_space,
            num_agents=5,
            communication_dim=32
        )
        print(f"   ✅ MultiAgentPolicy criada: {policy.num_agents} agentes")

        # Testar predição
        obs, info = env.reset()
        action = policy.predict(obs)[0]
        print(f"   ✅ Ação predita: {action.shape}")

        # Testar CentralizedPolicy
        print("\n2. Testando CentralizedPolicy...")
        from src.agents.policies import CentralizedPolicy

        policy = CentralizedPolicy(
            env.observation_space,
            env.action_space,
            num_buildings=5
        )
        print(f"   ✅ CentralizedPolicy criada: {policy.num_buildings} prédios")

        # Testar predição
        action = policy.predict(obs)[0]
        print(f"   ✅ Ação predita: {action.shape}")

        env.close()
        print("\n✅ Todas as políticas testadas com sucesso!")
        return True

    except Exception as e:
        print(f"❌ Erro nas políticas: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Teste específico para agentes SB3...")

    success1 = test_sb3_agents()
    success2 = test_policies()

    if success1 and success2:
        print("\n🎉 Todos os testes SB3 passaram!")
        print("✅ Os agentes com Stable Baselines3 estão funcionando!")
    else:
        print("\n⚠️ Alguns testes SB3 falharam")
        print("Verifique os erros acima")