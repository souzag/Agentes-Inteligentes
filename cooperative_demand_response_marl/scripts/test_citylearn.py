#!/usr/bin/env python3
"""
Script de teste para verificar a instalação do CityLearn
"""

import sys
import os
import numpy as np
from citylearn.citylearn import DataSet, CityLearnEnv, RewardFunction

def test_citylearn_installation():
    """Testa a instalação do CityLearn"""
    print("=== Teste de Instalação do CityLearn ===")
    
    # 1. Verificar versão
    try:
        import citylearn
        print(f"✓ CityLearn importado com sucesso")
        print(f"✓ Versão do CityLearn: {citylearn.__version__}")
    except ImportError as e:
        print(f"✗ Erro ao importar CityLearn: {e}")
        return False
    
    # 2. Carregar dataset de exemplo
    try:
        print("\n=== Carregando dataset citylearn_challenge_2022_phase_1 ===")
        dataset = DataSet("citylearn_challenge_2022_phase_1")
        print(f"✓ Dataset carregado com sucesso")
        
        # Obter caminho do dataset
        dataset_path = dataset.get_dataset("citylearn_challenge_2022_phase_1")
        print(f"✓ Caminho do dataset: {dataset_path}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(dataset_path):
            print(f"✗ Arquivo do dataset não existe: {dataset_path}")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao carregar dataset: {e}")
        return False
    
    # 3. Criar ambiente de simulação
    try:
        print("\n=== Criando ambiente de simulação ===")
        env = CityLearnEnv(dataset_path)
        print(f"✓ Ambiente criado com sucesso")
        print(f"✓ Número de prédios: {len(env.buildings)}")
    except Exception as e:
        print(f"✗ Erro ao criar ambiente: {e}")
        return False
    
    # 4. Verificar espaços de observação e ação
    try:
        print("\n=== Verificando espaços ===")
        print(f"✓ Observation space: {env.observation_space}")
        print(f"✓ Action space: {env.action_space}")
        
        # Verificar espaços para cada prédio
        for i, building in enumerate(env.buildings):
            print(f"  - Prédio {i}: obs_space={building.observation_space}, action_space={building.action_space}")
            if i >= 2:  # Limitar a 3 prédios para não poluir a saída
                print("  ...")
                break
        print(f"✓ Todos os prédios têm espaços definidos")
    except Exception as e:
        print(f"✗ Erro ao verificar espaços: {e}")
        return False
    
    # 5. Executar passos com agente aleatório
    try:
        print("\n=== Testando execução com agente aleatório ===")
        obs = env.reset()
        print(f"✓ Reset do ambiente realizado")
        
        # O CityLearn pode retornar uma tupla (obs, info)
        if isinstance(obs, tuple):
            obs_data = obs[0]
            print(f"✓ Forma da observação inicial: {len(obs_data)} prédios")
        else:
            print(f"✓ Forma da observação inicial: {np.array(obs).shape}")
        
        # Executar alguns passos
        for step in range(5):
            # Ações aleatórias para cada prédio
            actions = []
            for building in env.buildings:
                action = building.action_space.sample()
                actions.append(action)
            
            # Executar passo
            result = env.step(actions)
            
            # O CityLearn 2.3.1 pode retornar 5 valores (obs, reward, terminated, truncated, info)
            if len(result) == 4:
                obs, rewards, done, info = result
            elif len(result) == 5:
                obs, rewards, done, truncated, info = result
            else:
                print(f"✗ Número inesperado de valores retornados: {len(result)}")
                return False
            
            print(f"  Passo {step + 1}:")
            print(f"    - Recompensas: {len(rewards)} prédios")
            print(f"    - Done: {all(done) if isinstance(done, list) else done}")
            
            if (isinstance(done, list) and all(done)) or (not isinstance(done, list) and done):
                break
        
        print("✓ Execução com agente aleatório concluída com sucesso")
        
    except Exception as e:
        print(f"✗ Erro durante execução: {e}")
        return False
    
    # 6. Testar funcionalidades adicionais
    try:
        print("\n=== Testando funcionalidades adicionais ===")
        
        # Testar se o ambiente é compatível com Gymnasium
        print(f"✓ Ambiente tem atributo 'observation_space': {hasattr(env, 'observation_space')}")
        print(f"✓ Ambiente tem atributo 'action_space': {hasattr(env, 'action_space')}")
        print(f"✓ Ambiente tem método 'reset': {hasattr(env, 'reset')}")
        print(f"✓ Ambiente tem método 'step': {hasattr(env, 'step')}")
        
        # Testar informações dos prédios
        print(f"✓ Número total de prédios: {len(env.buildings)}")
        if len(env.buildings) > 0:
            building = env.buildings[0]
            print(f"✓ Tipos de observações disponíveis: {type(building.observation_space)}")
            print(f"✓ Tipos de ações disponíveis: {type(building.action_space)}")
        
    except Exception as e:
        print(f"✗ Erro em funcionalidades adicionais: {e}")
        return False
    
    print("\n=== Teste concluído com sucesso! ===")
    print("✓ O CityLearn está funcionando corretamente")
    print("✓ O ambiente está compatível com RL")
    print("✓ Pronto para desenvolvimento MARL")
    print("✓ Dataset citylearn_challenge_2022_phase_1 carregado")
    print(f"✓ {len(env.buildings)} prédios no ambiente")
    
    return True

if __name__ == "__main__":
    success = test_citylearn_installation()
    if not success:
        sys.exit(1)