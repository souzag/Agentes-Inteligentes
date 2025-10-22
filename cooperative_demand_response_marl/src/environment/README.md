# Ambiente Vetorizado CityLearn para MARL Cooperativo

## Visão Geral

Este diretório contém a implementação do ambiente vetorizado customizado para o CityLearn, otimizado para treinamento de algoritmos MARL (Multi-Agent Reinforcement Learning) cooperativos usando Stable Baselines3.

## Arquitetura

### CityLearnVecEnv

Classe principal que implementa um ambiente vetorizado multi-agente para o CityLearn:

```python
class CityLearnVecEnv(gymnasium.Env):
    """
    Ambiente vetorizado customizado para CityLearn com suporte a MARL cooperativo.

    Características:
    - Multi-agente: Cada prédio é um agente independente
    - Vetorizado: Compatível com Stable Baselines3
    - Cooperativo: Recompensas globais para incentivar cooperação
    - Flexível: Suporte a diferentes datasets e configurações
    """
```

## Componentes Principais

### 1. Espaços de Observação e Ação

- **Observation Space**: Concatenação das observações de todos os prédios
  - Dimensão: (n_buildings * 28,) para 28 features por prédio
  - Features: temporais, energéticas, econômicas, climáticas, do prédio

- **Action Space**: Ações de todos os prédios
  - Dimensão: (n_buildings,) para 1 ação por prédio
  - Range: [-0.78125, 0.78125] (normalizado)

### 2. Sistema de Recompensas

Implementa função de recompensa cooperativa que considera:

- **Objetivo Local**: Conforto térmico e custos individuais
- **Objetivo Global**: Balanceamento da rede e redução de picos
- **Cooperação**: Penalidades por ações não-coordenadas

### 3. Comunicação entre Agentes

- **Estado Global**: Todos os agentes têm acesso ao estado completo
- **Informações Compartilhadas**: Preços, demanda total, geração coletiva
- **Coordenação**: Mecanismos para sincronização de ações

## Implementação

### Arquivos

1. **`citylearn_vec_env.py`**: Implementação principal do ambiente
2. **`rewards.py`**: Funções de recompensa cooperativas
3. **`communication.py`**: Sistema de comunicação entre agentes
4. **`wrappers.py`**: Wrappers utilitários para compatibilidade

### Exemplo de Uso

```python
from src.environment.citylearn_vec_env import CityLearnVecEnv

# Criar ambiente
env = CityLearnVecEnv(
    dataset_name="citylearn_challenge_2022_phase_1",
    reward_function="cooperative",
    communication=True
)

# Usar com Stable Baselines3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## Configuração

### Parâmetros Principais

- `dataset_name`: Nome do dataset CityLearn
- `reward_function`: Tipo de recompensa ("local", "global", "cooperative")
- `communication`: Ativar comunicação entre agentes
- `normalize`: Normalizar observações
- `max_timesteps`: Limitar duração do episódio

### Datasets Suportados

- citylearn_challenge_2022_phase_1 (5 prédios)
- citylearn_challenge_2022_phase_2 (5 prédios)
- citylearn_challenge_2022_phase_3 (7 prédios)
- citylearn_challenge_2023_phase_1 (dataset disponível)
- citylearn_challenge_2023_phase_2 (dataset disponível)
- citylearn_challenge_2023_phase_3 (dataset disponível)

## Compatibilidade

### Stable Baselines3

- ✅ **VecEnv**: Compatível com DummyVecEnv e SubprocVecEnv
- ✅ **Gymnasium**: Interface padrão gymnasium.Env
- ✅ **Multi-Agent**: Suporte nativo a múltiplos agentes
- ✅ **Callbacks**: Suporte a callbacks de treinamento

### Algoritmos Recomendados

1. **PPO**: Para políticas estocásticas
2. **SAC**: Para ações contínuas
3. **MADDPG**: Para cooperação explícita
4. **MAPPO**: Para multi-agente com PPO

## Testes

### Testes Unitários

- Teste de criação do ambiente
- Teste de reset e step
- Teste de espaços de observação/ação
- Teste de funções de recompensa

### Testes de Integração

- Teste com Stable Baselines3
- Teste de treinamento completo
- Teste de diferentes datasets
- Teste de comunicação entre agentes

## Desempenho

### Métricas de Avaliação

- **Consumo Total**: kWh por episódio
- **Pico de Demanda**: Máximo consumo horário
- **Custos Operacionais**: Total em $
- **Conforto Térmico**: Desvio de temperatura
- **Emissões de CO2**: kg por episódio

### Benchmarks

Comparação com baselines:
- Agente aleatório
- Controle rule-based
- Agentes independentes (sem cooperação)
- Agentes centralizados

## Extensões Futuras

### Funcionalidades Planejadas

1. **Hierarchical RL**: Agentes com diferentes níveis de decisão
2. **Transfer Learning**: Reutilização de políticas entre datasets
3. **Meta-Learning**: Adaptação rápida a novos prédios
4. **Safe RL**: Garantias de segurança e conforto

### Integrações

- **EnergyPlus**: Simulação mais detalhada
- **OpenAI Gym**: Compatibilidade expandida
- **Ray RLlib**: Algoritmos distribuídos
- **Weights & Biases**: Tracking de experimentos

## Referências

- [CityLearn Documentation](https://intelligent-environments-lab.github.io/CityLearn/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Multi-Agent RL Papers](https://arxiv.org/abs/2011.00583)
- [Cooperative Demand Response](https://arxiv.org/abs/2003.04246)

## Contribuição

Para contribuir com melhorias:

1. Implementar novas funções de recompensa
2. Adicionar datasets adicionais
3. Otimizar performance
4. Expandir documentação

## Licença

Este código segue a mesma licença do projeto principal.