# Configuração dos Agentes MARL

## Visão Geral

Este documento descreve as configurações disponíveis para os agentes MARL no sistema de demand response cooperativo. As configurações são definidas em arquivos YAML e controlam o comportamento, aprendizado e comunicação dos agentes.

## Configuração Principal

### Arquivo de Configuração Base

```yaml
# config/agents.yaml
agents:
  # Tipo de agente
  type: "cooperative"  # "independent", "cooperative", "centralized"

  # Configurações gerais
  num_agents: 5
  seed: 42

  # Configurações de política
  policy:
    name: "MultiAgentPolicy"
    shared_parameters: true
    communication_dim: 32
    net_arch: [256, 256, 128]
    activation_fn: "ReLU"

  # Configurações de treinamento
  training:
    algorithm: "PPO"
    total_timesteps: 1000000
    eval_freq: 10000
    save_freq: 50000
    learning_rate: 3e-4
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01

  # Configurações de comunicação
  communication:
    enabled: true
    protocol: "full"  # "full", "neighborhood", "centralized", "hierarchical"
    message_dim: 16
    update_freq: 1

  # Configurações de recompensa
  reward:
    type: "cooperative"
    local_weight: 0.4
    global_weight: 0.4
    cooperation_weight: 0.2

  # Configurações de exploração
  exploration:
    initial_eps: 1.0
    final_eps: 0.1
    decay_steps: 100000
    noise_std: 0.1

  # Configurações de logging
  logging:
    enabled: true
    log_freq: 1000
    metrics: ["reward", "loss", "entropy", "communication"]
    tensorboard: true
    save_model: true
```

## Tipos de Agentes

### 1. Agente Independente

```yaml
# config/agents_independent.yaml
agents:
  type: "independent"

  policy:
    name: "MlpPolicy"  # Política padrão do SB3
    net_arch: [64, 64]
    activation_fn: "Tanh"

  training:
    algorithm: "PPO"
    total_timesteps: 500000
    learning_rate: 3e-4

  communication:
    enabled: false  # Sem comunicação

  reward:
    type: "local"  # Apenas recompensa local
```

### 2. Agente Cooperativo

```yaml
# config/agents_cooperative.yaml
agents:
  type: "cooperative"

  policy:
    name: "MultiAgentPolicy"
    shared_parameters: true
    communication_dim: 32
    net_arch: [256, 256, 128]
    activation_fn: "ReLU"

  training:
    algorithm: "PPO"
    total_timesteps: 1000000
    learning_rate: 3e-4
    n_steps: 2048
    batch_size: 64

  communication:
    enabled: true
    protocol: "full"
    message_dim: 16

  reward:
    type: "cooperative"
    local_weight: 0.4
    global_weight: 0.4
    cooperation_weight: 0.2
```

### 3. Agente Centralizado

```yaml
# config/agents_centralized.yaml
agents:
  type: "centralized"

  policy:
    name: "CentralizedPolicy"
    net_arch: [512, 256, 128]
    activation_fn: "ReLU"

  training:
    algorithm: "PPO"
    total_timesteps: 1000000
    learning_rate: 3e-4
    n_steps: 2048
    batch_size: 128

  communication:
    enabled: false  # Controle centralizado

  reward:
    type: "global"  # Recompensa global
```

## Configurações de Algoritmos

### PPO (Proximal Policy Optimization)

```yaml
# config/algorithms/ppo.yaml
algorithm: "PPO"

# Hiperparâmetros do PPO
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
clip_range_vf: null
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
use_sde: false
sde_sample_freq: -1
target_kl: 0.01
tensorboard_log: null
create_eval_env: false
policy_kwargs: {}
verbose: 1
seed: null
device: "auto"
_init_setup_model: true
```

### SAC (Soft Actor-Critic)

```yaml
# config/algorithms/sac.yaml
algorithm: "SAC"

# Hiperparâmetros do SAC
learning_rate: 3e-4
batch_size: 256
gamma: 0.99
tau: 0.005
train_freq: 1
gradient_steps: 1
action_noise: null
optimize_memory_usage: false
ent_coef: "auto"
target_update_interval: 1
target_entropy: "auto"
use_sde: false
sde_sample_freq: -1
use_sde_at_warmup: false
tensorboard_log: null
create_eval_env: false
policy_kwargs: {}
verbose: 1
seed: null
device: "auto"
```

### MADDPG (Multi-Agent DDPG)

```yaml
# config/algorithms/maddpg.yaml
algorithm: "MADDPG"

# Hiperparâmetros do MADDPG
actor_lr: 1e-3
critic_lr: 1e-3
gamma: 0.95
tau: 0.01
batch_size: 256
memory_size: 1000000
update_freq: 100
train_freq: 1
gradient_steps: 1
policy_delay: 2
target_policy_noise: 0.2
target_noise_clip: 0.5
tensorboard_log: null
create_eval_env: false
policy_kwargs: {}
verbose: 1
seed: null
```

## Configurações de Comunicação

### Comunicação Completa

```yaml
# config/communication/full.yaml
communication:
  enabled: true
  protocol: "full"

  # Configurações de mensagens
  message_dim: 16
  update_freq: 1
  compression: false

  # Tipos de mensagens
  message_types:
    - "state"
    - "action"
    - "reward"
    - "gradient"
```

### Comunicação por Vizinhança

```yaml
# config/communication/neighborhood.yaml
communication:
  enabled: true
  protocol: "neighborhood"

  # Parâmetros espaciais
  max_distance: 50.0
  positions: null  # Se null, usa distribuição aleatória

  # Configurações de mensagens
  message_dim: 8
  update_freq: 2
  compression: true

  # Topologia
  topology: "geographic"
  connectivity_threshold: 0.7
```

### Comunicação Centralizada

```yaml
# config/communication/centralized.yaml
communication:
  enabled: true
  protocol: "centralized"

  # Configurações do coordenador
  coordinator_id: 0
  coordination_strategy: "consensus"  # "consensus", "auction", "leader"

  # Configurações de mensagens
  message_dim: 32
  update_freq: 1
  compression: false

  # Broadcasting
  broadcast_freq: 10
  global_state_sharing: true
```

## Configurações de Recompensa

### Recompensa Local

```yaml
# config/rewards/local.yaml
reward:
  type: "local"

  # Componentes da recompensa
  comfort:
    weight: 0.5
    temperature_range: [20.0, 26.0]
    penalty_factor: 1.0

  cost:
    weight: 0.3
    electricity_pricing: true
    peak_penalty: 0.1

  efficiency:
    weight: 0.2
    storage_bonus: 0.01
    renewable_bonus: 0.02
```

### Recompensa Global

```yaml
# config/rewards/global.yaml
reward:
  type: "global"

  # Componentes globais
  peak_demand:
    weight: 0.4
    penalty_factor: 0.1
    baseline: 1000  # kW

  carbon_emissions:
    weight: 0.3
    emission_factor: 0.4  # kg CO2/kWh
    carbon_tax: 0.05  # $/kg CO2

  load_balancing:
    weight: 0.2
    target_load_factor: 0.8
    bonus_factor: 0.1

  grid_stability:
    weight: 0.1
    ramp_rate_penalty: 0.01
    voltage_penalty: 0.01
```

### Recompensa Cooperativa

```yaml
# config/rewards/cooperative.yaml
reward:
  type: "cooperative"

  # Pesos dos componentes
  local_weight: 0.4
  global_weight: 0.4
  cooperation_weight: 0.2

  # Componentes de cooperação
  coordination:
    weight: 0.5
    action_similarity_bonus: 0.1
    communication_bonus: 0.05

  resource_sharing:
    weight: 0.3
    storage_sharing_bonus: 0.02
    load_balancing_bonus: 0.03

  common_objective:
    weight: 0.2
    cost_reduction_bonus: 0.1
    efficiency_improvement_bonus: 0.05
```

## Configurações de Dataset

### Dataset Específico

```yaml
# config/datasets/citylearn_challenge_2022_phase_1.yaml
dataset:
  name: "citylearn_challenge_2022_phase_1"
  buildings: 5
  features_per_building: 28

  # Configurações específicas do dataset
  climate_zone: "Mixed-Humid"
  simulation_period: "1 year"
  time_step: "1 hour"

  # Normalização
  normalization:
    enabled: true
    method: "z_score"  # "z_score", "min_max", "robust"
    clip_range: [-5.0, 5.0]

  # Pré-processamento
  preprocessing:
    remove_outliers: true
    fill_missing: "interpolate"
    smoothing: false
```

## Configurações de Avaliação

### Métricas de Avaliação

```yaml
# config/evaluation/metrics.yaml
evaluation:
  metrics:
    # Métricas de energia
    energy:
      - "total_consumption"
      - "peak_demand"
      - "load_factor"
      - "energy_efficiency"

    # Métricas de conforto
    comfort:
      - "temperature_violations"
      - "comfort_satisfaction"
      - "thermal_comfort_index"

    # Métricas econômicas
    economic:
      - "total_cost"
      - "cost_savings"
      - "payback_period"

    # Métricas de cooperação
    cooperation:
      - "coordination_index"
      - "communication_efficiency"
      - "collective_performance"

  # Configurações de avaliação
  num_eval_episodes: 10
  eval_freq: 10000
  deterministic: true
  render: false
```

### Baselines para Comparação

```yaml
# config/evaluation/baselines.yaml
baselines:
  - name: "random_agent"
    type: "random"
    description: "Agente que seleciona ações aleatórias"

  - name: "rule_based"
    type: "rule_based"
    description: "Controlador baseado em regras fixas"
    rules:
      temperature_control: "hysteresis"
      storage_management: "greedy"

  - name: "independent_ppo"
    type: "independent"
    description: "Agentes PPO independentes"
    algorithm: "PPO"
    training_steps: 500000

  - name: "centralized_ppo"
    type: "centralized"
    description: "Agente PPO centralizado"
    algorithm: "PPO"
    training_steps: 1000000
```

## Configurações de Hardware

### Configuração para GPU

```yaml
# config/hardware/gpu.yaml
hardware:
  device: "cuda"  # "cpu", "cuda", "auto"
  num_gpus: 1
  gpu_memory_fraction: 0.9

  # Paralelização
  parallel:
    enabled: true
    num_envs: 8
    vec_env_type: "SubprocVecEnv"

  # Otimizações
  optimizations:
    mixed_precision: true
    gradient_checkpointing: false
    model_parallelism: false
```

### Configuração para CPU

```yaml
# config/hardware/cpu.yaml
hardware:
  device: "cpu"
  num_threads: 4

  # Paralelização
  parallel:
    enabled: true
    num_envs: 4
    vec_env_type: "DummyVecEnv"

  # Otimizações
  optimizations:
    mixed_precision: false
    gradient_checkpointing: false
    model_parallelism: false
```

## Configurações de Logging

### TensorBoard

```yaml
# config/logging/tensorboard.yaml
logging:
  tensorboard:
    enabled: true
    log_dir: "logs/tensorboard/"
    update_freq: 100

    # Métricas a logar
    scalars:
      - "train/reward"
      - "train/loss"
      - "train/entropy"
      - "eval/mean_reward"
      - "eval/std_reward"

    histograms:
      - "policy/action_dist"
      - "value_function/values"

    graphs:
      - "policy/network"
      - "value_function/network"
```

### Weights & Biases

```yaml
# config/logging/wandb.yaml
logging:
  wandb:
    enabled: true
    project: "citylearn_marl"
    entity: "your_entity"
    name: "cooperative_experiment"

    # Configurações
    config:
      algorithm: "PPO"
      num_agents: 5
      reward_type: "cooperative"

    # Métricas
    metrics:
      - "episode_reward"
      - "episode_length"
      - "energy_consumption"
      - "comfort_violations"
```

## Exemplos de Configuração Completa

### Configuração para Desenvolvimento

```yaml
# config/development.yaml
agents:
  type: "cooperative"
  num_agents: 5

  policy:
    name: "MultiAgentPolicy"
    shared_parameters: true
    communication_dim: 16  # Menor para desenvolvimento
    net_arch: [128, 128, 64]

  training:
    algorithm: "PPO"
    total_timesteps: 100000  # Menos para desenvolvimento
    eval_freq: 10000
    save_freq: 25000
    learning_rate: 3e-4

  communication:
    enabled: true
    protocol: "full"
    message_dim: 8

  reward:
    type: "cooperative"
    local_weight: 0.4
    global_weight: 0.4
    cooperation_weight: 0.2

  logging:
    enabled: true
    log_freq: 1000
    tensorboard: true
```

### Configuração para Produção

```yaml
# config/production.yaml
agents:
  type: "cooperative"
  num_agents: 7

  policy:
    name: "CooperativePolicy"
    shared_parameters: true
    communication_dim: 32
    net_arch: [256, 256, 128, 64]

  training:
    algorithm: "PPO"
    total_timesteps: 10000000  # Muito mais para produção
    eval_freq: 100000
    save_freq: 500000
    learning_rate: 3e-4

  communication:
    enabled: true
    protocol: "neighborhood"  # Mais realista
    message_dim: 16

  reward:
    type: "cooperative"
    local_weight: 0.3
    global_weight: 0.5
    cooperation_weight: 0.2

  hardware:
    device: "cuda"
    parallel:
      enabled: true
      num_envs: 16

  logging:
    enabled: true
    wandb: true
    tensorboard: true
```

## Validação de Configuração

### Função de Validação

```python
def validate_agent_config(config: Dict) -> bool:
    """Valida configuração dos agentes"""
    required_fields = [
        "agents.type",
        "agents.policy.name",
        "agents.training.algorithm"
    ]

    for field in required_fields:
        if not get_nested(config, field):
            raise ValueError(f"Campo obrigatório ausente: {field}")

    # Validar tipo de agente
    valid_agent_types = ["independent", "cooperative", "centralized"]
    if config["agents"]["type"] not in valid_agent_types:
        raise ValueError(f"Tipo de agente inválido: {config['agents']['type']}")

    # Validar algoritmo
    valid_algorithms = ["PPO", "SAC", "MADDPG", "A2C"]
    if config["agents"]["training"]["algorithm"] not in valid_algorithms:
        raise ValueError(f"Algoritmo inválido: {config['agents']['training']['algorithm']}")

    # Validar comunicação
    if config["agents"].get("communication", {}).get("enabled", False):
        valid_protocols = ["full", "neighborhood", "centralized", "hierarchical"]
        protocol = config["agents"]["communication"]["protocol"]
        if protocol not in valid_protocols:
            raise ValueError(f"Protocolo de comunicação inválido: {protocol}")

    return True
```

### Carregamento de Configuração

```python
def load_agent_config(config_path: str) -> Dict:
    """Carrega configuração de arquivo YAML"""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    validate_agent_config(config)
    return config

def create_config_from_dict(config_dict: Dict) -> Dict:
    """Cria configuração a partir de dicionário"""
    validate_agent_config(config_dict)
    return config_dict
```

## Templates de Configuração

### Template Básico

```python
def get_basic_config():
    """Retorna configuração básica para desenvolvimento"""
    return {
        "agents": {
            "type": "independent",
            "num_agents": 5,
            "policy": {
                "name": "MlpPolicy",
                "net_arch": [64, 64]
            },
            "training": {
                "algorithm": "PPO",
                "total_timesteps": 100000
            },
            "communication": {
                "enabled": False
            },
            "reward": {
                "type": "local"
            }
        }
    }
```

### Template Avançado

```python
def get_advanced_config():
    """Retorna configuração avançada para produção"""
    return {
        "agents": {
            "type": "cooperative",
            "num_agents": 7,
            "policy": {
                "name": "CooperativePolicy",
                "shared_parameters": True,
                "communication_dim": 32,
                "net_arch": [256, 256, 128]
            },
            "training": {
                "algorithm": "PPO",
                "total_timesteps": 10000000,
                "eval_freq": 100000,
                "save_freq": 500000
            },
            "communication": {
                "enabled": True,
                "protocol": "neighborhood",
                "message_dim": 16
            },
            "reward": {
                "type": "cooperative",
                "local_weight": 0.3,
                "global_weight": 0.5,
                "cooperation_weight": 0.2
            }
        }
    }
```

## Troubleshooting

### Problemas Comuns

1. **Configuração inválida**: Usar validate_agent_config()
2. **Recursos insuficientes**: Verificar configurações de hardware
3. **Convergência lenta**: Ajustar hiperparâmetros de treinamento
4. **Exploração insuficiente**: Aumentar ent_coef ou usar ruído

### Debugging

```python
# Verificar configuração carregada
config = load_agent_config("config/agents.yaml")
print("Configuração carregada:")
print(yaml.dump(config, default_flow_style=False))

# Verificar validação
try:
    validate_agent_config(config)
    print("✅ Configuração válida")
except ValueError as e:
    print(f"❌ Erro de validação: {e}")