# Especificações Técnicas - CityLearnVecEnv

## Visão Geral

O `CityLearnVecEnv` é uma implementação customizada de ambiente vetorizado para o CityLearn, otimizada para treinamento de algoritmos MARL (Multi-Agent Reinforcement Learning) cooperativos usando o framework Stable Baselines3.

## Interface Principal

### Classe CityLearnVecEnv

```python
class CityLearnVecEnv(gymnasium.Env):
    """
    Ambiente vetorizado multi-agente para CityLearn.

    Args:
        dataset_name (str): Nome do dataset CityLearn
        reward_function (str): Tipo de função de recompensa
        communication (bool): Ativar comunicação entre agentes
        normalize_obs (bool): Normalizar observações
        max_episode_length (int): Comprimento máximo do episódio
        seed (int): Seed para reprodutibilidade

    Attributes:
        num_buildings (int): Número de prédios/agentes
        observation_space (gymnasium.spaces.Box): Espaço de observação
        action_space (gymnasium.spaces.Box): Espaço de ação
        buildings (list): Lista de objetos Building do CityLearn
        reward_function (callable): Função de recompensa
    """
```

## Espaços de Observação e Ação

### Observation Space

```python
@property
def observation_space(self):
    """
    Retorna o espaço de observação do ambiente.

    Returns:
        gymnasium.spaces.Box: Espaço de observação

    Shape: (num_buildings * features_per_building,)
    Features per building: 28 (conforme análise do CityLearn)
    """
    # Para 5 prédios: (140,) - 5 * 28 features
    low = np.concatenate([building.observation_space.low for building in self.buildings])
    high = np.concatenate([building.observation_space.high for building in self.buildings])

    return gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
```

### Action Space

```python
@property
def action_space(self):
    """
    Retorna o espaço de ação do ambiente.

    Returns:
        gymnasium.spaces.Box: Espaço de ação

    Shape: (num_buildings,)
    Range: [-0.78125, 0.78125] por ação
    """
    # Para 5 prédios: (5,) - 1 ação por prédio
    low = np.array([-0.78125] * self.num_buildings)
    high = np.array([0.78125] * self.num_buildings)

    return gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
```

## Métodos Principais

### reset()

```python
def reset(self, **kwargs):
    """
    Reseta o ambiente para um novo episódio.

    Args:
        **kwargs: Argumentos adicionais para o reset

    Returns:
        tuple: (observations, info)
            observations (np.ndarray): Observações iniciais
            info (dict): Informações adicionais
    """
    # Reset do ambiente CityLearn
    citylearn_obs = self.citylearn_env.reset(**kwargs)

    # Converter para formato vetorizado
    if isinstance(citylearn_obs, tuple):
        obs, info = citylearn_obs
    else:
        obs, info = citylearn_obs, {}

    # Concatenar observações de todos os prédios
    vectorized_obs = self._concatenate_observations(obs)

    # Aplicar normalização se habilitada
    if self.normalize_obs:
        vectorized_obs = self._normalize_observations(vectorized_obs)

    return vectorized_obs, info
```

### step()

```python
def step(self, actions):
    """
    Executa um passo no ambiente.

    Args:
        actions (np.ndarray): Ações para todos os prédios

    Returns:
        tuple: (observations, rewards, dones, infos)
            observations (np.ndarray): Próximas observações
            rewards (np.ndarray): Recompensas para cada prédio
            dones (bool or array): Flags de término
            infos (dict): Informações adicionais
    """
    # Distribuir ações para os prédios
    building_actions = self._split_actions(actions)

    # Executar passo no CityLearn
    citylearn_result = self.citylearn_env.step(building_actions)

    # Processar resultado
    if len(citylearn_result) == 4:
        obs, rewards, done, info = citylearn_result
    else:
        obs, rewards, done, truncated, info = citylearn_result

    # Converter para formato vetorizado
    vectorized_obs = self._concatenate_observations(obs)

    # Aplicar função de recompensa customizada
    if self.reward_function:
        rewards = self.reward_function(obs, building_actions, info)

    # Aplicar normalização se habilitada
    if self.normalize_obs:
        vectorized_obs = self._normalize_observations(vectorized_obs)

    return vectorized_obs, rewards, done, info
```

## Funções Auxiliares

### _concatenate_observations()

```python
def _concatenate_observations(self, observations):
    """
    Concatena observações de todos os prédios em um vetor único.

    Args:
        observations: Observações do CityLearn (lista ou array)

    Returns:
        np.ndarray: Observações vetorizadas
    """
    if isinstance(observations, list):
        # Lista de arrays - um por prédio
        return np.concatenate([np.array(obs) for obs in observations])
    else:
        # Array único - converter para formato correto
        return np.array(observations).flatten()
```

### _split_actions()

```python
def _split_actions(self, actions):
    """
    Divide ações vetorizadas em ações individuais por prédio.

    Args:
        actions (np.ndarray): Ações vetorizadas

    Returns:
        list: Lista de ações por prédio
    """
    # actions shape: (num_buildings,)
    # Retorna lista com uma ação por prédio
    return [actions[i] for i in range(self.num_buildings)]
```

### _normalize_observations()

```python
def _normalize_observations(self, observations):
    """
    Normaliza observações usando estatísticas pré-computadas.

    Args:
        observations (np.ndarray): Observações não normalizadas

    Returns:
        np.ndarray: Observações normalizadas
    """
    if not hasattr(self, 'obs_mean') or not hasattr(self, 'obs_std'):
        # Inicializar estatísticas se não existirem
        self.obs_mean = np.zeros_like(observations)
        self.obs_std = np.ones_like(observations)

    # Normalização z-score
    normalized = (observations - self.obs_mean) / (self.obs_std + 1e-8)
    return normalized
```

## Sistema de Recompensas

### RewardFunction Base

```python
class RewardFunction:
    """Classe base para funções de recompensa"""

    def __init__(self, weights=None):
        self.weights = weights or {"comfort": 0.25, "cost": 0.25, "peak": 0.25, "cooperation": 0.25}

    def __call__(self, observations, actions, info):
        """Calcula recompensas para todos os prédios"""
        raise NotImplementedError

class CooperativeReward(RewardFunction):
    """Função de recompensa cooperativa"""

    def __call__(self, observations, actions, info):
        # Calcular componente local (conforto + custo)
        local_rewards = self._local_reward(observations, actions)

        # Calcular componente global (pico + emissões)
        global_reward = self._global_reward(observations, actions)

        # Calcular bônus de cooperação
        cooperation_bonus = self._cooperation_bonus(actions, info)

        # Combinar componentes
        total_reward = (self.weights["comfort"] * local_rewards +
                       self.weights["cost"] * (-info.get("cost", 0)) +
                       self.weights["peak"] * (-info.get("peak", 0)) +
                       self.weights["cooperation"] * cooperation_bonus)

        return total_reward
```

## Sistema de Comunicação

### CommunicationProtocol

```python
class CommunicationProtocol:
    """Protocolo de comunicação entre agentes"""

    def __init__(self, communication_type="full"):
        self.type = communication_type
        self.message_queue = []

    def send_message(self, sender_id, receiver_id, message):
        """Envia mensagem entre agentes"""
        if self.type == "full":
            # Todos podem se comunicar com todos
            self.message_queue.append((sender_id, receiver_id, message))
        elif self.type == "neighborhood":
            # Apenas vizinhos podem se comunicar
            if self._are_neighbors(sender_id, receiver_id):
                self.message_queue.append((sender_id, receiver_id, message))

    def receive_messages(self, agent_id):
        """Recebe mensagens para um agente"""
        agent_messages = [msg for sender, receiver, msg in self.message_queue
                         if receiver == agent_id]
        # Limpar mensagens processadas
        self.message_queue = [(s, r, m) for s, r, m in self.message_queue
                             if r != agent_id]
        return agent_messages
```

## Compatibilidade com Stable Baselines3

### VecEnv Interface

```python
def get_attr(self, attr_name):
    """Obtém atributo de todos os ambientes (para VecEnv)"""
    return getattr(self.citylearn_env, attr_name)

def set_attr(self, attr_name, value):
    """Define atributo em todos os ambientes (para VecEnv)"""
    setattr(self.citylearn_env, attr_name, value)

def env_method(self, method_name, *args, **kwargs):
    """Chama método em todos os ambientes (para VecEnv)"""
    return getattr(self.citylearn_env, method_name)(*args, **kwargs)

def get_images(self):
    """Obtém imagens de renderização (para VecEnv)"""
    return [self.render(mode="rgb_array")]
```

### Callbacks Support

```python
class CityLearnCallback(BaseCallback):
    """Callback customizado para logging específico do CityLearn"""

    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self):
        """Chamado a cada passo de treinamento"""
        if self.n_calls % self.eval_freq == 0:
            # Log de métricas específicas
            self.logger.record("energy/total_consumption",
                             self.training_env.get_total_consumption())
            self.logger.record("energy/peak_demand",
                             self.training_env.get_peak_demand())
            self.logger.record("comfort/violations",
                             self.training_env.get_comfort_violations())
```

## Configuração e Inicialização

### Factory Function

```python
def make_citylearn_vec_env(dataset_name="citylearn_challenge_2022_phase_1",
                          reward_function="cooperative",
                          communication=True,
                          normalize=True,
                          **kwargs):
    """
    Factory function para criar ambiente CityLearnVecEnv

    Args:
        dataset_name (str): Nome do dataset
        reward_function (str): Tipo de recompensa
        communication (bool): Ativar comunicação
        normalize (bool): Normalizar observações
        **kwargs: Argumentos adicionais

    Returns:
        CityLearnVecEnv: Ambiente configurado
    """
    # Importar CityLearn
    from citylearn.citylearn import DataSet, CityLearnEnv

    # Carregar dataset
    dataset = DataSet(dataset_name)
    dataset_path = dataset.get_dataset(dataset_name)

    # Criar ambiente base
    base_env = CityLearnEnv(dataset_path)

    # Criar wrapper vetorizado
    vec_env = CityLearnVecEnv(
        citylearn_env=base_env,
        reward_function=reward_function,
        communication=communication,
        normalize=normalize,
        **kwargs
    )

    return vec_env
```

## Testes Unitários

### Teste de Criação

```python
def test_citylearn_vec_env_creation():
    """Testa criação do ambiente"""
    env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")

    assert env.num_buildings == 5
    assert env.observation_space.shape == (140,)  # 5 * 28
    assert env.action_space.shape == (5,)  # 5 buildings
    assert env.action_space.low[0] == -0.78125
    assert env.action_space.high[0] == 0.78125

def test_reset_and_step():
    """Testa reset e step do ambiente"""
    env = make_citylearn_vec_env()

    # Teste reset
    obs, info = env.reset()
    assert obs.shape == (140,)
    assert isinstance(info, dict)

    # Teste step
    actions = env.action_space.sample()
    obs, rewards, done, info = env.step(actions)

    assert obs.shape == (140,)
    assert len(rewards) == 5  # 5 buildings
    assert isinstance(done, (bool, np.ndarray))
    assert isinstance(info, dict)
```

## Performance Benchmarks

### Métricas de Performance

```python
def benchmark_performance():
    """Benchmark de performance do ambiente"""
    import time

    env = make_citylearn_vec_env()

    # Benchmark reset
    start_time = time.time()
    for _ in range(100):
        env.reset()
    reset_time = time.time() - start_time

    # Benchmark step
    env.reset()
    actions = env.action_space.sample()

    start_time = time.time()
    for _ in range(1000):
        env.step(actions)
    step_time = time.time() - start_time

    print(f"Reset time (100 calls): {reset_time:.4f}s")
    print(f"Step time (1000 calls): {step_time:.4f}s")
    print(f"Steps per second: {1000/step_time:.2f}")
```

### Otimizações de Performance

1. **Vectorized Operations**: Usar numpy para operações vetorizadas
2. **Pre-allocation**: Pre-alocar arrays para evitar realocações
3. **Caching**: Cache de cálculos frequentes
4. **Lazy Loading**: Carregamento lazy de datasets

## Documentação de API

### Parâmetros de Inicialização

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| dataset_name | str | "citylearn_challenge_2022_phase_1" | Nome do dataset CityLearn |
| reward_function | str | "cooperative" | Tipo de função de recompensa |
| communication | bool | True | Ativar comunicação entre agentes |
| normalize_obs | bool | True | Normalizar observações |
| max_episode_length | int | 8760 | Comprimento máximo do episódio |
| seed | int | None | Seed para reprodutibilidade |

### Métodos Públicos

| Método | Retorno | Descrição |
|--------|---------|-----------|
| reset() | tuple | Reseta ambiente e retorna observações iniciais |
| step(actions) | tuple | Executa passo e retorna (obs, rewards, done, info) |
| render(mode) | None | Renderiza estado do ambiente |
| close() | None | Fecha ambiente e libera recursos |

### Atributos

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| num_buildings | int | Número de prédios no ambiente |
| observation_space | gymnasium.spaces.Box | Espaço de observação |
| action_space | gymnasium.spaces.Box | Espaço de ação |
| buildings | list | Lista de objetos Building |
| reward_function | callable | Função de recompensa atual |

## Exemplos de Uso

### Uso Básico

```python
from src.environment.citylearn_vec_env import make_citylearn_vec_env

# Criar ambiente
env = make_citylearn_vec_env(
    dataset_name="citylearn_challenge_2022_phase_1",
    reward_function="cooperative",
    communication=True
)

# Usar com Stable Baselines3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Uso Avançado

```python
# Configuração customizada
env = make_citylearn_vec_env(
    dataset_name="citylearn_challenge_2022_phase_3",
    reward_function="cooperative",
    communication=True,
    normalize=True,
    max_episode_length=1000  # Para testes rápidos
)

# Treinamento com callbacks
from stable_baselines3.common.callbacks import EvalCallback

eval_env = make_citylearn_vec_env(dataset_name="citylearn_challenge_2022_phase_1")
eval_callback = EvalCallback(eval_env, eval_freq=1000)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, callback=eval_callback)
```

## Troubleshooting

### Problemas Comuns

1. **ImportError**: CityLearn não instalado
   ```bash
   pip install citylearn
   ```

2. **Dataset não encontrado**: Cache do CityLearn
   ```python
   from citylearn.citylearn import DataSet
   DataSet.clear_cache()  # Limpar cache
   ```

3. **Espaços incorretos**: Verificar dataset
   ```python
   # Verificar número de prédios
   env = CityLearnEnv(dataset_path)
   print(f"Buildings: {len(env.buildings)}")
   ```

4. **Performance lenta**: Usar SubprocVecEnv
   ```python
   from stable_baselines3.common.vec_env import SubprocVecEnv

   def make_env():
       return make_citylearn_vec_env()

   env = SubprocVecEnv([make_env for _ in range(4)])
   ```

### Debugging

```python
# Ativar logging detalhado
import logging
logging.basicConfig(level=logging.DEBUG)

# Verificar espaços
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Testar reset e step
obs, info = env.reset()
print(f"Initial obs shape: {obs.shape}")

actions = env.action_space.sample()
obs, rewards, done, info = env.step(actions)
print(f"Rewards: {rewards}")
print(f"Done: {done}")