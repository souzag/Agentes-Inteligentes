#!/usr/bin/env python3
"""
Script simples para testar a API do CityLearn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from citylearn.citylearn import DataSet, CityLearnEnv, RewardFunction

def test_simple():
    """Teste simples da API do CityLearn"""
    print("=== Teste Simples do CityLearn ===")
    
    # Testar dataset
    dataset = DataSet('citylearn_challenge_2022_phase_1')
    print(f"✓ Dataset criado: {dataset}")
    
    # Obter dados do dataset
    dataset_path = dataset.get_dataset('citylearn_challenge_2022_phase_1')
    print(f"✓ Caminho do dataset: {dataset_path}")
    
    # Verificar se o arquivo existe
    if os.path.exists(dataset_path):
        print("✓ Arquivo do dataset existe")
        
        # Tentar criar ambiente diretamente com o caminho do dataset
        try:
            env = CityLearnEnv(dataset_path)
            print("✓ Ambiente criado com sucesso!")
            print(f"✓ Número de prédios: {len(env.buildings)}")
            print(f"✓ Observation space: {env.observation_space}")
            print(f"✓ Action space: {env.action_space}")
            
            # Testar reset
            obs = env.reset()
            if isinstance(obs, tuple):
                print(f"✓ Reset realizado, forma da observação: {len(obs)} elementos")
                print(f"✓ Tipo de observação: tupla")
            else:
                print(f"✓ Reset realizado, forma da observação: {obs.shape}")
            
            # Testar um passo
            actions = [building.action_space.sample() for building in env.buildings]
            obs, rewards, done, info = env.step(actions)
            print(f"✓ Passo executado, recompensas: {len(rewards)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Erro ao criar ambiente: {e}")
            return False
    else:
        print("✗ Arquivo do dataset não existe")
        return False

if __name__ == "__main__":
    success = test_simple()
    if not success:
        sys.exit(1)