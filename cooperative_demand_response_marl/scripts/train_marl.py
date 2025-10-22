#!/usr/bin/env python3
"""
Script de treinamento para agentes MARL no sistema de demand response.

Este script demonstra como treinar diferentes tipos de agentes MARL
(cooperativos, independentes, centralizados) no ambiente CityLearn,
usando Stable Baselines3 e os componentes implementados.
"""

import os
import sys
import numpy as np
import yaml
from typing import Dict, List, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Configura o ambiente e importa dependências."""
    try:
        # Adicionar diretório src ao path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Importar componentes
        from src.environment import make_citylearn_vec_env
        from src.agents import (
            AgentFactory,
            MultiAgentFactory,
            IndependentAgentFactory,
            CooperativeAgentFactory,
            CentralizedAgentFactory,
            RandomAgentFactory,
            RuleBasedAgentFactory
        )
        from src.communication import (
            FullCommunication,
            NeighborhoodCommunication,
            CentralizedCommunication,
            HierarchicalCommunication
        )

        print("✅ Ambiente configurado com sucesso")
        return {
            'make_citylearn_vec_env': make_citylearn_vec_env,
            'AgentFactory': AgentFactory,
            'MultiAgentFactory': MultiAgentFactory,
            'IndependentAgentFactory': IndependentAgentFactory,
            'CooperativeAgentFactory': CooperativeAgentFactory,
            'CentralizedAgentFactory': CentralizedAgentFactory,
            'RandomAgentFactory': RandomAgentFactory,
            'RuleBasedAgentFactory': RuleBasedAgentFactory,
            'FullCommunication': FullCommunication,
            'NeighborhoodCommunication': NeighborhoodCommunication,
            'CentralizedCommunication': CentralizedCommunication,
            'HierarchicalCommunication': HierarchicalCommunication
        }

    except ImportError as e:
        print(f"❌ Erro ao importar dependências: {e}")
        print("Certifique-se de que todas as dependências estão instaladas:")
        print("  pip install stable-baselines3 gymnasium citylearn")
        sys.exit(1)

def load_config(config_path: str) -> Dict:
    """Carrega configuração de arquivo YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuração carregada de {config_path}")
        return config
    except Exception as e:
        print(f"⚠️ Erro ao carregar configuração: {e}")
        print("Usando configuração padrão...")
        return get_default_config()

def get_default_config() -> Dict:
    """Retorna configuração padrão para treinamento."""
    return {
        "environment": {
            "dataset": "citylearn_challenge_2022_phase_1",
            "reward_function": "cooperative"
        },
        "agents": {
            "type": "cooperative",
            "num_agents": 5,
            "communication": {
                "enabled": True,
                "protocol": "full"
            }
        },
        "training": {
            "algorithm": "PPO",
            "total_timesteps": 100000,
            "eval_freq": 10000,
            "save_freq": 50000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64
        },
        "logging": {
            "tensorboard": True,
            "save_models": True,
            "log_dir": "logs/training/"
        }
    }

def create_environment(config: Dict):
    """Cria ambiente baseado na configuração."""
    env_config = config["environment"]

    print(f"🏗️ Criando ambiente: {env_config['dataset']}")

    env = config['make_citylearn_vec_env'](
        dataset_name=env_config["dataset"],
        reward_function=env_config.get("reward_function", "cooperative")
    )

    print(f"✅ Ambiente criado: {env.num_buildings} prédios")
    return env

def create_communication_protocol(config: Dict):
    """Cria protocolo de comunicação baseado na configuração."""
    comm_config = config["agents"].get("communication", {})

    if not comm_config.get("enabled", False):
        return None

    protocol_type = comm_config.get("protocol", "full")
    num_agents = config["agents"].get("num_agents", 5)

    print(f"💬 Criando protocolo de comunicação: {protocol_type}")

    if protocol_type == "full":
        protocol = config['FullCommunication'](num_agents)
    elif protocol_type == "neighborhood":
        protocol = config['NeighborhoodCommunication'](num_agents)
    elif protocol_type == "centralized":
        protocol = config['CentralizedCommunication'](num_agents)
    elif protocol_type == "hierarchical":
        protocol = config['HierarchicalCommunication'](num_agents)
    else:
        protocol = config['FullCommunication'](num_agents)

    print(f"✅ Protocolo criado: {type(protocol).__name__}")
    return protocol

def create_agents(env, config: Dict, communication_protocol=None):
    """Cria agentes baseado na configuração."""
    agent_config = config["agents"]
    agent_type = agent_config["type"]

    print(f"🤖 Criando agentes: {agent_type}")

    if agent_type == "independent":
        agents = config['IndependentAgentFactory'].create_multi_agent_system(env)
    elif agent_type == "cooperative":
        agents = config['CooperativeAgentFactory'].create_multi_agent_system(
            env, communication_protocol
        )
    elif agent_type == "centralized":
        agent = config['CentralizedAgentFactory'].create_centralized_system(env)
        agents = [agent]
    elif agent_type == "random":
        agents = config['RandomAgentFactory'].create_multi_agent_system(env)
    elif agent_type == "rule_based":
        agents = config['RuleBasedAgentFactory'].create_multi_agent_system(env)
    else:
        # Fallback para agentes independentes
        agents = config['IndependentAgentFactory'].create_multi_agent_system(env)

    print(f"✅ Agentes criados: {len(agents)} agentes do tipo {agent_type}")
    return agents

def train_agents(agents: List, env, config: Dict):
    """Treina os agentes."""
    training_config = config["training"]

    print(f"🏋️ Iniciando treinamento...")
    print(f"   - Algoritmo: {training_config['algorithm']}")
    print(f"   - Passos: {training_config['total_timesteps']}")
    print(f"   - Avaliação a cada: {training_config['eval_freq']} passos")

    # Configurar logging
    log_config = config.get("logging", {})
    if log_config.get("tensorboard", False):
        from stable_baselines3.common.callbacks import EvalCallback
        from src.environment import make_citylearn_vec_env

        eval_env = make_citylearn_vec_env(config["environment"]["dataset"])
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=training_config["eval_freq"],
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        callbacks = [eval_callback]
    else:
        callbacks = []

    # Treinar cada agente
    for i, agent in enumerate(agents):
        print(f"\n📈 Treinando agente {i} ({agent.__class__.__name__})...")

        try:
            agent.train(
                total_timesteps=training_config["total_timesteps"],
                eval_freq=training_config["eval_freq"],
                callback=callbacks if i == 0 else None  # Apenas um callback
            )

            # Salvar modelo se configurado
            if log_config.get("save_models", False):
                model_dir = log_config.get("log_dir", "models/trained/")
                os.makedirs(model_dir, exist_ok=True)

                model_path = os.path.join(model_dir, f"agent_{i}_{agent.__class__.__name__}.zip")
                agent.save_model(model_path)
                print(f"💾 Modelo salvo: {model_path}")

        except Exception as e:
            print(f"❌ Erro no treinamento do agente {i}: {e}")
            continue

    print("✅ Treinamento concluído!")

def evaluate_agents(agents: List, env, config: Dict):
    """Avalia performance dos agentes."""
    eval_config = config.get("evaluation", {})
    num_episodes = eval_config.get("num_episodes", 10)

    print(f"📊 Avaliando agentes por {num_episodes} episódios...")

    results = {}

    for i, agent in enumerate(agents):
        print(f"\n🔍 Avaliando agente {i} ({agent.__class__.__name__})...")

        try:
            eval_result = agent.evaluate(num_episodes=num_episodes)
            results[f"agent_{i}"] = eval_result

            print(f"   - Recompensa média: {eval_result['mean_reward']:.3f}")
            print(f"   - Desvio padrão: {eval_result['std_reward']:.3f}")
            print(f"   - Recompensa min/máx: {eval_result['min_reward']:.3f} / {eval_result['max_reward']:.3f}")

        except Exception as e:
            print(f"❌ Erro na avaliação do agente {i}: {e}")
            continue

    # Salvar resultados
    results_dir = "results/evaluation/"
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "agent_evaluation.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"💾 Resultados salvos em {results_path}")
    return results

def compare_with_baselines(env, config: Dict):
    """Compara agentes treinados com baselines."""
    print("\n🏆 Comparando com baselines...")

    # Criar baselines
    baselines = config['MultiAgentFactory'].create_baseline_agents(env)

    # Avaliar baselines
    baseline_results = {}
    for name, agents in baselines.items():
        print(f"\n📊 Avaliando baseline: {name}")

        if isinstance(agents, list):
            # Múltiplos agentes
            agent_results = []
            for i, agent in enumerate(agents):
                result = agent.evaluate(num_episodes=5)  # Menos episódios para baselines
                agent_results.append(result)
                print(f"   Agente {i}: {result['mean_reward']:.3f}")

            # Agregar resultados
            mean_rewards = [r['mean_reward'] for r in agent_results]
            baseline_results[name] = {
                "mean_reward": np.mean(mean_rewards),
                "std_reward": np.std(mean_rewards),
                "num_agents": len(agents)
            }
        else:
            # Agente único
            result = agents.evaluate(num_episodes=5)
            baseline_results[name] = result

    # Salvar comparação
    comparison_dir = "results/comparison/"
    os.makedirs(comparison_dir, exist_ok=True)

    comparison_path = os.path.join(comparison_dir, "baseline_comparison.yaml")
    with open(comparison_path, 'w') as f:
        yaml.dump(baseline_results, f, default_flow_style=False)

    print(f"💾 Comparação salva em {comparison_path}")
    return baseline_results

def run_experiment(config_path: Optional[str] = None, experiment_name: str = "default"):
    """Executa experimento completo de treinamento MARL."""
    print("=" * 60)
    print(f"EXPERIMENTO MARL: {experiment_name}")
    print("=" * 60)

    # Setup
    components = setup_environment()

    # Carregar configuração
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()
        print("⚠️ Usando configuração padrão")

    # Adicionar componentes à configuração
    config.update(components)

    # Criar ambiente
    env = create_environment(config)

    # Criar comunicação
    comm_protocol = create_communication_protocol(config)

    # Criar agentes
    agents = create_agents(env, config, comm_protocol)

    # Treinar agentes
    train_agents(agents, env, config)

    # Avaliar agentes
    eval_results = evaluate_agents(agents, env, config)

    # Comparar com baselines
    baseline_results = compare_with_baselines(env, config)

    # Fechar ambiente
    env.close()

    print("\n" + "=" * 60)
    print("EXPERIMENTO CONCLUÍDO!")
    print("=" * 60)
    print(f"📁 Resultados salvos em: results/evaluation/")
    print(f"📁 Modelos salvos em: {config['logging'].get('log_dir', 'models/trained/')}")
    print(f"📊 Comparação com baselines: results/comparison/")

    return {
        "config": config,
        "eval_results": eval_results,
        "baseline_results": baseline_results,
        "experiment_name": experiment_name
    }

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Treinamento de agentes MARL para CityLearn")
    parser.add_argument("--config", "-c", type=str, help="Caminho do arquivo de configuração YAML")
    parser.add_argument("--experiment", "-e", type=str, default="default", help="Nome do experimento")
    parser.add_argument("--quick", "-q", action="store_true", help="Execução rápida para testes")

    args = parser.parse_args()

    # Configuração rápida para testes
    if args.quick:
        config = get_default_config()
        config["training"]["total_timesteps"] = 10000  # Muito menos para teste rápido
        config["evaluation"] = {"num_episodes": 3}
        print("🚀 Modo rápido ativado!")

        # Executar experimento rápido
        result = run_experiment(config_path=None, experiment_name=f"{args.experiment}_quick")

    else:
        # Executar experimento completo
        result = run_experiment(config_path=args.config, experiment_name=args.experiment)

    print(f"\n🎯 Experimento '{result['experiment_name']}' concluído com sucesso!")

if __name__ == "__main__":
    main()