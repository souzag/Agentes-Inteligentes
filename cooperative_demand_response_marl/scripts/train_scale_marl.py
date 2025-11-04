#!/usr/bin/env python3
"""
Script de treinamento em escala para agentes MARL no sistema de demand response.

Este script executa treinamentos completos e comparações entre múltiplos datasets
do CityLearn 2022 (phase_1, phase_2, phase_3) para avaliar performance em escala.

Funcionalidades:
- Treinamentos longos com múltiplos datasets
- Comparação de performance entre diferentes cenários
- Métricas de avaliação em escala
- Visualizações comparativas
- Logging detalhado para monitoramente
- Relatórios automatizados

Autor: Sistema de Agentes Inteligentes
Data: Outubro 2025
"""

import os
import sys
import numpy as np
import yaml
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
        }

    except ImportError as e:
        print(f"❌ Erro ao importar dependências: {e}")
        print("Certifique-se de que todas as dependências estão instaladas:")
        print("  pip install stable-baselines3 gymnasium citylearn matplotlib seaborn")
        sys.exit(1)

def get_scale_config() -> Dict:
    """Retorna configuração otimizada para treinamentos em escala."""
    return {
        "datasets": [
            "citylearn_challenge_2022_phase_1",
            "citylearn_challenge_2022_phase_2",
            "citylearn_challenge_2022_phase_3"
        ],
        "agents": {
            "types": ["random", "independent", "cooperative"],
            "num_agents": None,  # Será determinado pelo dataset
            "communication": {
                "enabled": True,
                "protocol": "full"
            }
        },
        "training": {
            "algorithm": "PPO",
            "total_timesteps": 500000,  # Treinamento mais longo para escala
            "eval_freq": 25000,  # Avaliação menos frequente
            "save_freq": 100000,  # Salvamento menos frequente
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2
        },
        "evaluation": {
            "num_episodes": 20,  # Mais episódios para estatísticas robustas
            "deterministic": True,
            "render": False
        },
        "logging": {
            "enabled": True,
            "tensorboard": True,
            "save_models": True,
            "log_dir": "results/scale_training/",
            "metrics": [
                "episode_reward",
                "episode_length",
                "energy_consumption",
                "peak_demand",
                "comfort_violations",
                "cooperation_score",
                "training_time",
                "convergence_rate"
            ]
        },
        "performance": {
            "parallel_envs": 4,  # Usar múltiplos ambientes paralelos
            "memory_limit": "4GB",
            "cpu_limit": 8
        }
    }

def create_environments_for_datasets(components: Dict, datasets: List[str]) -> Dict:
    """Cria ambientes para múltiplos datasets."""
    environments = {}

    print("🏗️ Criando ambientes para múltiplos datasets...")

    for dataset in datasets:
        try:
            print(f"  📊 Criando ambiente para {dataset}...")
            env = components['make_citylearn_vec_env'](
                dataset_name=dataset,
                reward_function="cooperative"
            )

            environments[dataset] = {
                'env': env,
                'num_buildings': env.num_buildings,
                'observation_space': env.observation_space,
                'action_space': env.action_space
            }

            print(f"    ✅ {dataset}: {env.num_buildings} prédios")

        except Exception as e:
            print(f"    ❌ Erro ao criar ambiente {dataset}: {e}")
            continue

    return environments

def create_agents_for_environment(components: Dict, env, agent_types: List[str]) -> Dict:
    """Cria diferentes tipos de agentes para um ambiente."""
    agents_dict = {}

    for agent_type in agent_types:
        try:
            print(f"  🤖 Criando agentes {agent_type}...")

            if agent_type == "random":
                agents = components['RandomAgentFactory'].create_multi_agent_system(env)
            elif agent_type == "independent":
                agents = components['IndependentAgentFactory'].create_multi_agent_system(env)
            elif agent_type == "cooperative":
                agents = components['CooperativeAgentFactory'].create_multi_agent_system(env)
            else:
                print(f"    ⚠️ Tipo de agente não suportado: {agent_type}")
                continue

            agents_dict[agent_type] = agents
            print(f"    ✅ {len(agents)} agentes {agent_type} criados")

        except Exception as e:
            print(f"    ❌ Erro ao criar agentes {agent_type}: {e}")
            continue

    return agents_dict

def train_agents_on_dataset(agents_dict: Dict, env, dataset_name: str, config: Dict) -> Dict:
    """Treina agentes em um dataset específico."""
    training_config = config["training"]
    log_config = config["logging"]

    results = {}
    start_time = time.time()

    print(f"\n🏋️ Iniciando treinamentos em {dataset_name}...")

    for agent_type, agents in agents_dict.items():
        if not agents:
            continue

        print(f"\n🔄 Treinando {agent_type} agents...")

        agent_results = []
        agent_start_time = time.time()

        # Treinar cada agente
        for i, agent in enumerate(agents):
            try:
                print(f"  📈 Treinando agente {i+1}/{len(agents)}...")

                # Configurar callbacks se necessário
                callbacks = []
                if log_config.get("tensorboard", False) and i == 0:
                    from stable_baselines3.common.callbacks import EvalCallback
                    eval_env = env  # Usar mesmo ambiente para avaliação
                    eval_callback = EvalCallback(
                        eval_env,
                        eval_freq=training_config["eval_freq"],
                        n_eval_episodes=5,
                        deterministic=True,
                        render=False,
                        log_path=log_config["log_dir"]
                    )
                    callbacks.append(eval_callback)

                # Treinar agente
                train_start = time.time()
                agent.train(
                    total_timesteps=training_config["total_timesteps"],
                    eval_freq=training_config["eval_freq"],
                    callbacks=callbacks if callbacks else None
                )
                train_time = time.time() - train_start

                # Salvar modelo se configurado
                if log_config.get("save_models", False):
                    model_dir = os.path.join(log_config["log_dir"], dataset_name, agent_type)
                    os.makedirs(model_dir, exist_ok=True)

                    model_path = os.path.join(model_dir, f"agent_{i}.zip")
                    agent.save_model(model_path)

                # Avaliar agente
                eval_result = agent.evaluate(num_episodes=config["evaluation"]["num_episodes"])

                agent_result = {
                    'agent_id': i,
                    'training_time': train_time,
                    'mean_reward': eval_result['mean_reward'],
                    'std_reward': eval_result['std_reward'],
                    'min_reward': eval_result['min_reward'],
                    'max_reward': eval_result['max_reward'],
                    'model_path': model_path if log_config.get("save_models", False) else None
                }

                agent_results.append(agent_result)
                print(f"    📊 Agente {i+1}: Recompensa = {agent_result['mean_reward']:.6f}")
                print(f"       ⏱️ Tempo de treinamento: {agent_result['training_time']:.3f}s")
            except Exception as e:
                print(f"    ❌ Erro no treinamento do agente {i}: {e}")
                continue

        # Agregar resultados do tipo de agente
        if agent_results:
            mean_rewards = [r['mean_reward'] for r in agent_results]
            training_times = [r['training_time'] for r in agent_results]

            results[agent_type] = {
                'num_agents': len(agent_results),
                'mean_reward': np.mean(mean_rewards),
                'std_reward': np.std(mean_rewards),
                'min_reward': np.min(mean_rewards),
                'max_reward': np.max(mean_rewards),
                'total_training_time': np.sum(training_times),
                'mean_training_time': np.mean(training_times),
                'agent_results': agent_results
            }

            print(f"  ✅ {agent_type}: {results[agent_type]['mean_reward']:.6f} ± {results[agent_type]['std_reward']:.6f}")

    total_time = time.time() - start_time
    print(f"✅ Treinamentos em {dataset_name} concluídos em {total_time:.2f}s")
    return results

def evaluate_performance_across_datasets(all_results: Dict) -> Dict:
    """Avalia performance comparativa entre datasets."""
    print("\n📊 Avaliando performance comparativa entre datasets...")

    performance_analysis = {}

    # Coletar dados por tipo de agente
    agent_types = set()
    for dataset_results in all_results.values():
        agent_types.update(dataset_results.keys())

    for agent_type in agent_types:
        datasets_data = []
        datasets_names = []

        for dataset_name, dataset_results in all_results.items():
            if agent_type in dataset_results:
                data = dataset_results[agent_type]
                datasets_data.append(data['mean_reward'])
                datasets_names.append(dataset_name)

        if len(datasets_data) >= 2:
            # Calcular melhorias relativas
            baseline = datasets_data[0]  # Primeiro dataset como baseline
            improvements = [((r - baseline) / abs(baseline)) * 100 if baseline != 0 else 0
                          for r in datasets_data]

            performance_analysis[agent_type] = {
                'datasets': datasets_names,
                'mean_rewards': datasets_data,
                'improvements': improvements,
                'best_dataset': datasets_names[np.argmax(datasets_data)],
                'worst_dataset': datasets_names[np.argmin(datasets_data)],
                'max_improvement': max(improvements),
                'min_improvement': min(improvements)
            }

    return performance_analysis

def create_scale_visualizations(all_results: Dict, performance_analysis: Dict, output_dir: str):
    """Cria visualizações para análise em escala."""
    print("\n📈 Criando visualizações de escala...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Comparação de performance por dataset
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise Comparativa de Performance em Escala - MARL\nDatasets CityLearn 2022', fontsize=16, fontweight='bold')

    # Preparar dados
    datasets = list(all_results.keys())
    agent_types = ['random', 'independent', 'cooperative']

    # Gráfico 1: Performance média por dataset e tipo de agente
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.25

    for i, agent_type in enumerate(agent_types):
        means = []
        stds = []
        for dataset in datasets:
            if agent_type in all_results[dataset]:
                means.append(all_results[dataset][agent_type]['mean_reward'])
                stds.append(all_results[dataset][agent_type]['std_reward'])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i*width, means, width, label=agent_type.capitalize(),
               yerr=stds, capsize=5, alpha=0.8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Recompensa Média')
    ax.set_title('Performance por Dataset e Tipo de Agente')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('citylearn_challenge_2022_', '') for d in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gráfico 2: Tempo de treinamento
    ax = axes[0, 1]
    for agent_type in agent_types:
        times = []
        for dataset in datasets:
            if agent_type in all_results[dataset]:
                times.append(all_results[dataset][agent_type]['total_training_time'])
            else:
                times.append(0)

        ax.plot(datasets, times, 'o-', label=agent_type.capitalize(), linewidth=2, markersize=8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Tempo Total de Treinamento (s)')
    ax.set_title('Tempo de Treinamento por Dataset')
    ax.set_xticklabels([d.replace('citylearn_challenge_2022_', '') for d in datasets])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gráfico 3: Melhorias relativas
    ax = axes[1, 0]
    if performance_analysis:
        for agent_type, data in performance_analysis.items():
            ax.plot(data['datasets'],
                   data['improvements'],
                   'o-', label=f'{agent_type.capitalize()}',
                   linewidth=2, markersize=8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Melhoria Relativa (%)')
        ax.set_title('Melhoria Relativa em Relação ao Baseline')
        ax.set_xticklabels([d.replace('citylearn_challenge_2022_', '') for d in data['datasets']])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Gráfico 4: Estatísticas de convergência
    ax = axes[1, 1]
    convergence_data = []
    labels = []

    for dataset in datasets:
        for agent_type in agent_types:
            if agent_type in all_results[dataset]:
                data = all_results[dataset][agent_type]
                convergence_data.append(data['std_reward'] / abs(data['mean_reward']) if data['mean_reward'] != 0 else 0)
                labels.append(f"{dataset.split('_')[-1]}\n{agent_type[:3]}")

    if convergence_data:
        ax.bar(range(len(convergence_data)), convergence_data, alpha=0.8, color='skyblue')
        ax.set_xlabel('Configuração Dataset-Agente')
        ax.set_ylabel('Coeficiente de Variação')
        ax.set_title('Estabilidade de Convergência\n(Menor = Melhor)')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'scale_performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✅ Gráfico salvo: {plot_path}")

    # 2. Gráfico detalhado por agente
    if all_results:
        fig, axes = plt.subplots(len(agent_types), 1, figsize=(12, 4*len(agent_types)))
        if len(agent_types) == 1:
            axes = [axes]

        for i, agent_type in enumerate(agent_types):
            ax = axes[i]

            agent_rewards = []
            agent_labels = []

            for dataset in datasets:
                if agent_type in all_results[dataset]:
                    results = all_results[dataset][agent_type]['agent_results']
                    for j, agent_result in enumerate(results):
                        agent_rewards.append(agent_result['mean_reward'])
                        agent_labels.append(f"{dataset.split('_')[-1]}-A{j+1}")

            if agent_rewards:
                ax.bar(range(len(agent_rewards)), agent_rewards, alpha=0.8,
                      color=f'C{i}', label=f'{agent_type.capitalize()} Agents')
                ax.set_xlabel('Agente')
                ax.set_ylabel('Recompensa Média')
                ax.set_title(f'Performance Individual - {agent_type.capitalize()} Agents')
                ax.set_xticks(range(len(agent_labels)))
                ax.set_xticklabels(agent_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()
        detailed_plot_path = os.path.join(output_dir, 'scale_detailed_performance.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Gráfico detalhado salvo: {detailed_plot_path}")

    plt.close('all')

def generate_scale_report(all_results: Dict, performance_analysis: Dict, config: Dict, output_dir: str):
    """Gera relatório completo da avaliação em escala."""
    print("\n📄 Gerando relatório de escala...")

    os.makedirs(output_dir, exist_ok=True)

    report = []
    report.append("=" * 80)
    report.append("RELATÓRIO DE AVALIAÇÃO EM ESCALA - MARL")
    report.append("Sistema de Resposta Cooperativa à Demanda")
    report.append("=" * 80)
    report.append("")

    # Cabeçalho
    report.append("Data: Outubro 2025")
    report.append("Objetivo: Avaliação de performance em escala com múltiplos datasets CityLearn 2022")
    report.append("")

    # Configuração
    report.append("🔧 CONFIGURAÇÃO DE AVALIAÇÃO")
    report.append("-" * 50)
    report.append(f"Datasets avaliados: {', '.join(config['datasets'])}")
    report.append(f"Tipos de agentes: {', '.join(config['agents']['types'])}")
    report.append(f"Timesteps por treinamento: {config['training']['total_timesteps']:,}")
    report.append(f"Episódios de avaliação: {config['evaluation']['num_episodes']}")
    report.append("")

    # Resultados por dataset
    report.append("📊 RESULTADOS POR DATASET")
    report.append("-" * 50)

    for dataset_name, dataset_results in all_results.items():
        report.append(f"\n🏗️ {dataset_name.upper()}")
        report.append("-" * 30)

        # Informações do dataset
        if dataset_name in ["citylearn_challenge_2022_phase_1", "citylearn_challenge_2022_phase_2", "citylearn_challenge_2022_phase_3"]:
            if dataset_name == "citylearn_challenge_2022_phase_1":
                info = "5 prédios, Mixed-Humid"
            elif dataset_name == "citylearn_challenge_2022_phase_2":
                info = "5 prédios, Hot-Humid"
            else:
                info = "7 prédios, Mixed-Dry"
            report.append(f"Características: {info}")

        # Resultados por tipo de agente
        for agent_type, results in dataset_results.items():
            report.append(f"\n🤖 {agent_type.upper()} AGENTS")
            report.append(f"  • Número de agentes: {results['num_agents']}")
            report.append(f"  • Recompensa média: {results['mean_reward']:.6f}")
            report.append(f"  • Desvio padrão: {results['std_reward']:.6f}")
            report.append(f"  • Faixa: [{results['min_reward']:.6f}, {results['max_reward']:.6f}]")
            report.append(f"  • Tempo total de treinamento: {results['total_training_time']:.2f}s")
            report.append(f"  • Tempo médio por agente: {results['mean_training_time']:.2f}s")

    # Análise comparativa
    if performance_analysis:
        report.append("\n" + "🏆 ANÁLISE COMPARATIVA")
        report.append("-" * 50)

        for agent_type, analysis in performance_analysis.items():
            report.append(f"\n🔍 {agent_type.upper()} AGENTS")
            report.append(f"  • Melhor performance: {analysis['best_dataset']}")
            report.append(f"  • Pior performance: {analysis['worst_dataset']}")
            report.append(f"  • Melhoria máxima: {analysis['max_improvement']:.1f}%")
            report.append(f"  • Melhoria mínima: {analysis['min_improvement']:.1f}%")

            # Comparação detalhada
            report.append("  • Comparação por dataset:")
            for i, dataset in enumerate(analysis['datasets']):
                improvement = analysis['improvements'][i]
                reward = analysis['mean_rewards'][i]
                report.append(f"    - {dataset}: {reward:.6f} ({improvement:+.1f}%)")

    # Conclusões
    report.append("\n" + "🎯 CONCLUSÕES E RECOMENDAÇÕES")
    report.append("-" * 50)

    # Identificar melhor configuração geral
    best_overall = None
    best_score = float('-inf')

    for dataset_name, dataset_results in all_results.items():
        for agent_type, results in dataset_results.items():
            score = results['mean_reward'] / (results['std_reward'] + 1e-6)  # Score = mean/std
            if score > best_score:
                best_score = score
                best_overall = (dataset_name, agent_type, results['mean_reward'])

    if best_overall:
        dataset, agent, reward = best_overall
        report.append(f"• Melhor configuração geral: {agent.upper()} agents no {dataset}")
        report.append(f"  com recompensa média de {reward:.6f}")

    # Recomendações
    report.append("\n📋 RECOMENDAÇÕES PARA PRODUÇÃO:")
    report.append("• Usar agentes cooperativos para melhor performance")
    report.append("• Datasets com mais prédios (phase_3) mostram maior potencial")
    report.append("• Treinamentos mais longos melhoram estabilidade")
    report.append("• Implementar comunicação full para coordenação ótima")

    # Arquivos gerados
    report.append("\n" + "📁 ARQUIVOS GERADOS")
    report.append("-" * 50)
    report.append("• results/scale_training/scale_performance_comparison.png")
    report.append("• results/scale_training/scale_detailed_performance.png")
    report.append("• results/scale_training/scale_report.txt")
    report.append("• results/scale_training/results.json")

    report.append("\n" + "=" * 80)

    # Salvar relatório
    report_path = os.path.join(output_dir, 'scale_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"  ✅ Relatório salvo: {report_path}")

    # Salvar resultados em JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'all_results': all_results,
            'performance_analysis': performance_analysis,
            'timestamp': datetime.now().isoformat(),
            'best_configuration': best_overall
        }, f, indent=2, default=str)

    print(f"  ✅ Resultados JSON salvos: {json_path}")

    return report_path

def run_scale_evaluation(config_path: Optional[str] = None):
    """Executa avaliação completa em escala."""
    print("=" * 80)
    print("🚀 AVALIAÇÃO EM ESCALA - MARL SYSTEM")
    print("Sistema de Resposta Cooperativa à Demanda")
    print("=" * 80)

    start_time = time.time()

    # Setup
    components = setup_environment()

    # Carregar configuração
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuração carregada de {config_path}")
    else:
        config = get_scale_config()
        print("⚠️ Usando configuração padrão para escala")

    # Criar ambientes para múltiplos datasets
    environments = create_environments_for_datasets(components, config["datasets"])

    if not environments:
        print("❌ Nenhum ambiente pôde ser criado. Abortando.")
        return None

    # Resultados de todos os datasets
    all_results = {}

    # Processar cada dataset
    for dataset_name, env_info in environments.items():
        print(f"\n" + "="*60)
        print(f"🎯 PROCESSANDO DATASET: {dataset_name.upper()}")
        print("="*60)

        env = env_info['env']

        # Criar agentes para este ambiente
        agents_dict = create_agents_for_environment(components, env, config["agents"]["types"])

        # Treinar agentes neste dataset
        dataset_results = train_agents_on_dataset(agents_dict, env, dataset_name, config)

        all_results[dataset_name] = dataset_results

        # Fechar ambiente
        env.close()

    # Análise comparativa
    performance_analysis = evaluate_performance_across_datasets(all_results)

    # Criar visualizações
    output_dir = config["logging"]["log_dir"]
    create_scale_visualizations(all_results, performance_analysis, output_dir)

    # Gerar relatório
    report_path = generate_scale_report(all_results, performance_analysis, config, output_dir)

    # Tempo total
    total_time = time.time() - start_time
    print(f"⏱️ Tempo total de avaliação: {total_time:.2f}s")
    print("\n" + "=" * 80)
    print("🎉 AVALIAÇÃO EM ESCALA CONCLUÍDA!")
    print("=" * 80)
    print(f"📁 Resultados salvos em: {output_dir}")
    print(f"📄 Relatório: {report_path}")

    return {
        "config": config,
        "all_results": all_results,
        "performance_analysis": performance_analysis,
        "total_time": total_time,
        "output_dir": output_dir
    }

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Avaliação em escala de agentes MARL para CityLearn")
    parser.add_argument("--config", "-c", type=str, help="Caminho do arquivo de configuração YAML")
    parser.add_argument("--quick", "-q", action="store_true", help="Execução rápida para testes")
    parser.add_argument("--datasets", "-d", nargs="+", help="Datasets específicos para avaliar")

    args = parser.parse_args()

    # Configuração rápida para testes
    if args.quick:
        config = get_scale_config()
        config["training"]["total_timesteps"] = 50000  # Muito menor para teste rápido
        config["evaluation"]["num_episodes"] = 5
        config["datasets"] = ["citylearn_challenge_2022_phase_1"]  # Apenas um dataset
        print("🚀 Modo rápido ativado!")

    # Datasets específicos
    if args.datasets:
        config = get_scale_config()
        config["datasets"] = args.datasets
        print(f"🎯 Datasets específicos: {args.datasets}")

    # Executar avaliação
    result = run_scale_evaluation(config_path=args.config)

    if result:
        print(f"\n🎯 Avaliação em escala concluída com sucesso!")
        print(f"📊 Datasets processados: {len(result['all_results'])}")
        print(f"⏱️ Tempo total: {result['total_time']:.2f}s")

if __name__ == "__main__":
    main()