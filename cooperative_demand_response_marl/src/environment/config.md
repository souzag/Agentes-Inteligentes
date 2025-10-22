# Configuração do Ambiente Vetorizado CityLearn

## Parâmetros de Configuração

### Configuração Principal

```yaml
# src/environment/citylearn_vec_env.yaml
environment:
  name: "CityLearnVecEnv"
  dataset: "citylearn_challenge_2022_phase_1"

  # Configurações de recompensa
  reward:
    type: "cooperative"  # "local", "global", "cooperative"
    weights:
      comfort: 0.3
      cost: 0.3
      peak: 0.2
      cooperation: 0.2

  # Configurações de comunicação
  communication:
    enabled: true
    type: "full"  # "full", "neighborhood", "centralized"
    range: 100  # Para comunicação por vizinhança

  # Configurações de observação
  observation:
    normalize: true
    stack_frames: 1
    include_global_state: true

  # Configurações de ação
  action:
    clip: true
    noise: 0.0  # Para exploração

  # Configurações de episódio
  episode:
    max_timesteps: 8760  # Um ano
    random_start: false
    seed: 42
```

## Datasets Disponíveis

### citylearn_challenge_2022_phase_1
```yaml
dataset: "citylearn_challenge_2022_phase_1"
buildings: 5
features_per_building: 28
climate_zone: "Mixed-Humid"
simulation_period: "1 year"
```

### citylearn_challenge_2022_phase_2
```yaml
dataset: "citylearn_challenge_2022_phase_2"
buildings: 5
features_per_building: 28
climate_zone: "Hot-Humid"
simulation_period: "1 year"
```

### citylearn_challenge_2022_phase_3
```yaml
dataset: "citylearn_challenge_2022_phase_3"
buildings: 7
features_per_building: 28
climate_zone: "Mixed-Dry"
simulation_period: "1 year"
```

## Funções de Recompensa

### 1. Recompensa Local
```python
def local_reward(self, building_states, actions):
    """
    Recompensa baseada apenas no estado individual do prédio

    Componentes:
    - Conforto térmico: penalidade por desvio de temperatura
    - Custos energéticos: custo de eletricidade
    - Eficiência: uso eficiente de recursos
    """
    comfort_penalty = temperature_deviation_penalty(building_states)
    cost_penalty = electricity_cost_penalty(building_states, actions)
    efficiency_bonus = storage_efficiency_bonus(building_states)

    return - (comfort_penalty + cost_penalty) + efficiency_bonus
```

### 2. Recompensa Global
```python
def global_reward(self, all_building_states, actions):
    """
    Recompensa baseada no estado global da rede

    Componentes:
    - Balanceamento: redução de picos de demanda
    - Emissões: redução de carbono
    - Estabilidade: suavização da curva de carga
    """
    peak_penalty = peak_demand_penalty(all_building_states)
    carbon_penalty = carbon_emissions_penalty(all_building_states)
    load_factor_bonus = load_factor_improvement_bonus(all_building_states)

    return - (peak_penalty + carbon_penalty) + load_factor_bonus
```

### 3. Recompensa Cooperativa
```python
def cooperative_reward(self, building_states, actions, global_state):
    """
    Combinação de recompensas locais e globais com bônus de cooperação

    Componentes:
    - Recompensa local ponderada
    - Recompensa global ponderada
    - Bônus por coordenação entre agentes
    - Penalidade por ações conflitantes
    """
    local_r = self.local_reward(building_states, actions)
    global_r = self.global_reward(building_states, actions)
    cooperation_bonus = coordination_bonus(actions, global_state)

    return (self.w_local * local_r +
            self.w_global * global_r +
            self.w_coop * cooperation_bonus)
```

## Configurações de Comunicação

### Comunicação Completa
```yaml
communication:
  enabled: true
  type: "full"
  channels:
    - state_sharing: true
    - action_coordination: true
    - reward_sharing: true
```

### Comunicação por Vizinhança
```yaml
communication:
  enabled: true
  type: "neighborhood"
  range: 50  # metros
  topology: "geographic"
```

### Comunicação Centralizada
```yaml
communication:
  enabled: true
  type: "centralized"
  coordinator: "auction"  # "auction", "consensus", "leader"
```

## Configurações de Stable Baselines3

### PPO
```yaml
algorithm: "PPO"
policy: "MlpPolicy"
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
```

### SAC
```yaml
algorithm: "SAC"
policy: "MlpPolicy"
learning_rate: 3e-4
batch_size: 256
gamma: 0.99
tau: 0.005
train_freq: 1
gradient_steps: 1
```

### MADDPG
```yaml
algorithm: "MADDPG"
actor_lr: 1e-3
critic_lr: 1e-3
gamma: 0.95
tau: 0.01
batch_size: 256
memory_size: 1000000
```

## Configurações de Logging

```yaml
logging:
  enabled: true
  level: "INFO"
  metrics:
    - episode_reward
    - episode_length
    - energy_consumption
    - peak_demand
    - comfort_violations
    - cooperation_score

  visualization:
    enabled: true
    plots:
      - energy_flows
      - demand_curve
      - temperature_profiles
      - reward_components

  tensorboard:
    enabled: true
    log_dir: "logs/tensorboard/"
    update_freq: 100
```

## Configurações de Teste

```yaml
testing:
  enabled: true
  test_episodes: 10
  render: false

  baselines:
    - random_agent
    - rule_based_controller
    - independent_agents

  metrics:
    - total_energy_consumption
    - peak_demand_reduction
    - cost_savings
    - comfort_satisfaction
    - cooperation_index
```

## Configurações de Hardware

```yaml
hardware:
  num_cpu: 4
  num_gpu: 0
  memory_limit: "2GB"

  parallel:
    enabled: true
    num_envs: 8
    vec_env: "SubprocVecEnv"  # "DummyVecEnv" or "SubprocVecEnv"
```

## Configurações de Segurança

```yaml
safety:
  enabled: true
  constraints:
    temperature_bounds:
      min: 20.0  # Celsius
      max: 26.0
    power_limits:
      max_demand: 1000  # kW
      max_ramp_rate: 100  # kW/hour

  recovery:
    enabled: true
    safe_actions: true
    emergency_shutdown: true
```

## Exemplo de Configuração Completa

```yaml
# config/citylearn_vec_env.yaml
environment:
  name: "CityLearnVecEnv"
  dataset: "citylearn_challenge_2022_phase_1"

  reward:
    type: "cooperative"
    weights:
      comfort: 0.25
      cost: 0.25
      peak: 0.25
      cooperation: 0.25

  communication:
    enabled: true
    type: "full"

  observation:
    normalize: true
    include_global_state: true

  episode:
    max_timesteps: 8760
    random_start: false

algorithm:
  name: "PPO"
  policy: "MlpPolicy"
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64

logging:
  enabled: true
  tensorboard: true
  metrics: ["reward", "energy", "comfort", "peak"]

training:
  total_timesteps: 1000000
  eval_freq: 10000
  save_freq: 50000
  num_eval_episodes: 5
```

## Validação de Configuração

```python
def validate_config(config):
    """Valida configuração do ambiente"""
    required_fields = [
        "environment.name",
        "environment.dataset",
        "environment.reward.type",
        "algorithm.name"
    ]

    for field in required_fields:
        if not get_nested(config, field):
            raise ValueError(f"Missing required field: {field}")

    # Validar tipos de recompensa
    valid_reward_types = ["local", "global", "cooperative"]
    if config["environment"]["reward"]["type"] not in valid_reward_types:
        raise ValueError(f"Invalid reward type: {config['environment']['reward']['type']}")

    # Validar dataset
    valid_datasets = [
        "citylearn_challenge_2022_phase_1",
        "citylearn_challenge_2022_phase_2",
        "citylearn_challenge_2022_phase_3"
    ]
    if config["environment"]["dataset"] not in valid_datasets:
        raise ValueError(f"Invalid dataset: {config['environment']['dataset']}")

    return True
```

## Carregamento de Configuração

```python
def load_config(config_path):
    """Carrega configuração de arquivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config

def create_env_from_config(config):
    """Cria ambiente a partir de configuração"""
    env_config = config["environment"]
    algo_config = config["algorithm"]

    env = CityLearnVecEnv(
        dataset_name=env_config["dataset"],
        reward_type=env_config["reward"]["type"],
        communication=env_config.get("communication", {}).get("enabled", False)
    )

    return env, algo_config