# Políticas Customizadas para Agentes MARL

## Visão Geral

Este documento descreve as políticas customizadas implementadas para os agentes MARL no sistema de demand response cooperativo. As políticas são otimizadas para o ambiente CityLearn e compatíveis com Stable Baselines3.

## Políticas Implementadas

### 1. MultiAgentPolicy

Política base para múltiplos agentes com suporte a comunicação.

```python
class MultiAgentPolicy(BasePolicy):
    """
    Política para múltiplos agentes com comunicação opcional.

    Args:
        observation_space: Espaço de observação
        action_space: Espaço de ação
        num_agents: Número de agentes
        communication_dim: Dimensão do canal de comunicação
        shared_parameters: Se deve compartilhar parâmetros entre agentes
    """

    def __init__(self, observation_space, action_space, num_agents=5,
                 communication_dim=0, shared_parameters=True):
        super().__init__(observation_space, action_space)

        self.num_agents = num_agents
        self.communication_dim = communication_dim
        self.shared_parameters = shared_parameters

        # Rede neural compartilhada
        self.shared_network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Processamento de comunicação
        if communication_dim > 0:
            self.comm_network = nn.Sequential(
                nn.Linear(communication_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )

        # Cabeça da política
        input_dim = 128 + (32 if communication_dim > 0 else 0)
        self.action_network = nn.Linear(input_dim, action_space.shape[0])

        # Inicialização dos pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Inicialização customizada dos pesos"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor, communication: Optional[torch.Tensor] = None):
        """Forward pass da política"""
        # Processar observação
        features = self.shared_network(obs)

        # Processar comunicação se disponível
        if communication is not None and self.communication_dim > 0:
            comm_features = self.comm_network(communication)
            combined = torch.cat([features, comm_features], dim=-1)
        else:
            combined = features

        # Gerar ação
        action = self.action_network(combined)
        return action

    def predict(self, observation: np.ndarray, communication: Optional[np.ndarray] = None,
                deterministic: bool = True):
        """Prediz ação baseada na observação"""
        self.set_training_mode(False)

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        if communication is not None:
            comm_tensor = torch.tensor(communication, dtype=torch.float32).unsqueeze(0)
        else:
            comm_tensor = None

        with torch.no_grad():
            action = self.forward(obs_tensor, comm_tensor)

        action = action.squeeze(0).cpu().numpy()

        if deterministic:
            return action, None
        else:
            # Adicionar ruído para exploração
            noise = np.random.normal(0, 0.1, size=action.shape)
            return action + noise, None
```

### 2. CooperativePolicy

Política especificamente otimizada para cooperação entre agentes.

```python
class CooperativePolicy(MultiAgentPolicy):
    """
    Política otimizada para cooperação entre agentes.

    Inclui mecanismos específicos para:
    - Coordenação de ações
    - Compartilhamento de informações
    - Objetivos cooperativos
    """

    def __init__(self, observation_space, action_space, num_agents=5,
                 communication_dim=32, cooperation_strength=0.1):
        super().__init__(observation_space, action_space, num_agents, communication_dim)

        self.cooperation_strength = cooperation_strength

        # Camada adicional para processamento cooperativo
        self.cooperation_network = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh()  # Limitar valores entre -1 e 1
        )

        # Mecanismo de atenção para comunicação
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

    def forward(self, obs: torch.Tensor, communication: Optional[torch.Tensor] = None):
        """Forward pass com mecanismos cooperativos"""
        # Processamento base
        features = self.shared_network(obs)

        # Processar comunicação com atenção
        if communication is not None and self.communication_dim > 0:
            comm_features = self.comm_network(communication)

            # Aplicar atenção aos features de comunicação
            combined_features = torch.cat([features, comm_features], dim=-1)
            coop_features = self.cooperation_network(combined_features)

            # Mecanismo de atenção
            attention_output, _ = self.attention(coop_features.unsqueeze(0),
                                               coop_features.unsqueeze(0),
                                               coop_features.unsqueeze(0))
            attention_output = attention_output.squeeze(0)

            final_features = features + self.cooperation_strength * attention_output
        else:
            final_features = features

        # Gerar ação
        action = self.action_network(final_features)
        return action
```

### 3. CentralizedPolicy

Política para agente centralizado que controla todos os prédios.

```python
class CentralizedPolicy(BasePolicy):
    """
    Política centralizada para controle global de todos os prédios.

    Args:
        observation_space: Espaço de observação global
        action_space: Espaço de ação global
        num_buildings: Número de prédios controlados
    """

    def __init__(self, observation_space, action_space, num_buildings=5):
        super().__init__(observation_space, action_space)

        self.num_buildings = num_buildings

        # Rede neural para controle centralizado
        self.global_network = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Cabeças separadas para cada prédio
        self.building_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_buildings)
        ])

        # Inicialização
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Inicialização dos pesos"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """Forward pass para controle centralizado"""
        # Processar estado global
        global_features = self.global_network(obs)

        # Gerar ações para cada prédio
        actions = []
        for head in self.building_heads:
            action = head(global_features)
            actions.append(action)

        return torch.cat(actions, dim=-1)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Prediz ações para todos os prédios"""
        self.set_training_mode(False)

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            actions = self.forward(obs_tensor)

        actions = actions.squeeze(0).cpu().numpy()

        if deterministic:
            return actions, None
        else:
            # Adicionar ruído para exploração
            noise = np.random.normal(0, 0.1, size=actions.shape)
            return actions + noise, None
```

## Configurações de Políticas

### Configuração YAML

```yaml
# config/policies.yaml
policies:
  multi_agent:
    name: "MultiAgentPolicy"
    shared_parameters: true
    communication_dim: 32
    net_arch: [256, 256, 128]
    activation_fn: "ReLU"

  cooperative:
    name: "CooperativePolicy"
    shared_parameters: true
    communication_dim: 32
    cooperation_strength: 0.1
    attention_heads: 4
    net_arch: [256, 128, 64]

  centralized:
    name: "CentralizedPolicy"
    num_buildings: 5
    net_arch: [512, 256, 128]
    building_heads: 5
    activation_fn: "ReLU"
```

## Integração com Stable Baselines3

### Registro de Políticas Customizadas

```python
def register_custom_policies():
    """Registra políticas customizadas no SB3"""

    # Registrar MultiAgentPolicy
    from stable_baselines3.common.policies import register_policy

    register_policy("MultiAgentPolicy", MultiAgentPolicy)
    register_policy("CooperativePolicy", CooperativePolicy)
    register_policy("CentralizedPolicy", CentralizedPolicy)

    print("✅ Políticas customizadas registradas no SB3")
```

### Uso com Algoritmos

```python
# PPO com política customizada
from stable_baselines3 import PPO

model = PPO(
    policy="MultiAgentPolicy",
    env=env,
    policy_kwargs={
        "num_agents": 5,
        "communication_dim": 32,
        "shared_parameters": True
    },
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# SAC com política customizada
from stable_baselines3 import SAC

model = SAC(
    policy="CooperativePolicy",
    env=env,
    policy_kwargs={
        "num_agents": 5,
        "communication_dim": 32,
        "cooperation_strength": 0.1
    },
    learning_rate=3e-4,
    batch_size=256,
    verbose=1
)
```

## Callbacks Customizados

### TrainingCallback

```python
class MultiAgentCallback(BaseCallback):
    """Callback customizado para treinamento multi-agente"""

    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self):
        """Chamado a cada passo de treinamento"""
        if self.n_calls % self.eval_freq == 0:
            # Log de métricas específicas
            self.logger.record("train/total_reward", self.locals["rewards"].sum())
            self.logger.record("train/mean_reward", self.locals["rewards"].mean())
            self.logger.record("train/reward_std", self.locals["rewards"].std())

            # Log de comunicação se disponível
            if hasattr(self.training_env, 'communication_enabled'):
                self.logger.record("communication/messages_sent",
                                 self.training_env.get_message_count())

    def _on_rollout_end(self):
        """Chamado no final de cada rollout"""
        # Log de métricas do episódio
        if hasattr(self.locals, 'ep_info_buffer'):
            ep_info = self.locals['ep_info_buffer']
            if ep_info:
                self.logger.record("rollout/ep_reward_mean", np.mean([ep['r'] for ep in ep_info]))
                self.logger.record("rollout/ep_len_mean", np.mean([ep['l'] for ep in ep_info]))
```

## Testes de Políticas

### Teste de Funcionalidade

```python
def test_policy_functionality():
    """Testa funcionalidade das políticas"""
    from src.environment import make_citylearn_vec_env

    env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")

    # Testar MultiAgentPolicy
    policy = MultiAgentPolicy(
        env.observation_space,
        env.action_space,
        num_agents=5,
        communication_dim=32
    )

    obs, info = env.reset()
    action = policy.predict(obs)[0]

    assert env.action_space.contains(action), "Ação fora do espaço válido"
    print("✅ MultiAgentPolicy funcionando")

    # Testar CentralizedPolicy
    policy = CentralizedPolicy(
        env.observation_space,
        env.action_space,
        num_buildings=5
    )

    action = policy.predict(obs)[0]
    assert env.action_space.contains(action), "Ação fora do espaço válido"
    print("✅ CentralizedPolicy funcionando")
```

### Teste de Performance

```python
def benchmark_policies():
    """Benchmark de performance das políticas"""
    import time
    from src.environment import make_citylearn_vec_env

    env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")

    policies = {
        "MultiAgentPolicy": MultiAgentPolicy(env.observation_space, env.action_space, 5, 32),
        "CentralizedPolicy": CentralizedPolicy(env.observation_space, env.action_space, 5)
    }

    for name, policy in policies.items():
        print(f"\nBenchmarking {name}...")

        # Benchmark de inferência
        obs, info = env.reset()
        num_inferences = 1000

        start_time = time.time()
        for _ in range(num_inferences):
            policy.predict(obs)
        inference_time = time.time() - start_time

        print(f"   - {num_inferences} inferências: {inference_time:.3f}s")
        print(f"   - Tempo médio: {inference_time/num_inferences*1000:.2f}ms")
        print(f"   - Inferências/segundo: {num_inferences/inference_time:.1f}")
```

## Otimizações de Performance

### 1. Parameter Sharing

```python
def create_shared_policy(observation_space, action_space, num_agents):
    """Cria política com compartilhamento de parâmetros"""
    return MultiAgentPolicy(
        observation_space=observation_space,
        action_space=action_space,
        num_agents=num_agents,
        shared_parameters=True,  # Compartilhar parâmetros
        communication_dim=32
    )
```

### 2. Communication Compression

```python
def compress_communication(communication_state, compression_ratio=0.5):
    """Comprime estado de comunicação"""
    if communication_state is None:
        return None

    # Compressão simples usando PCA ou autoencoder
    compressed_dim = int(communication_state.shape[-1] * compression_ratio)

    # Implementação simplificada: truncar
    return communication_state[..., :compressed_dim]
```

### 3. Batch Processing

```python
def batch_predict_policies(policies, observations, communications=None):
    """Processa múltiplas políticas em lote"""
    batch_actions = []

    for i, policy in enumerate(policies):
        obs = observations[i] if isinstance(observations, list) else observations
        comm = communications[i] if communications is not None else None

        action = policy.predict(obs, comm)[0]
        batch_actions.append(action)

    return np.array(batch_actions)
```

## Exemplos de Uso Avançado

### Treinamento com Comunicação

```python
from src.environment import make_citylearn_vec_env
from src.communication import FullCommunication
from src.agents.policies import CooperativePolicy

# Criar ambiente e comunicação
env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")
comm_protocol = FullCommunication(env.num_buildings)

# Criar política cooperativa
policy = CooperativePolicy(
    env.observation_space,
    env.action_space,
    num_agents=env.num_buildings,
    communication_dim=32
)

# Treinamento com comunicação
from stable_baselines3 import PPO

model = PPO(
    policy=policy,
    env=env,
    learning_rate=3e-4,
    verbose=1
)

# Customizar ambiente para incluir comunicação
class CommunicationWrapper(gymnasium.Wrapper):
    def step(self, action):
        # Interceptar step para incluir comunicação
        obs, reward, done, info = self.env.step(action)

        # Adicionar comunicação se necessário
        if hasattr(self.env, 'comm_protocol'):
            # Processar comunicação entre agentes
            pass

        return obs, reward, done, info

env = CommunicationWrapper(env)
env.comm_protocol = comm_protocol

# Treinar
model.learn(total_timesteps=100000)
```

### Avaliação de Políticas

```python
def evaluate_policies(env, policies, num_episodes=10):
    """Avalia múltiplas políticas"""
    results = {}

    for name, policy in policies.items():
        print(f"Avaliando {name}...")

        episode_rewards = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = policy.predict(obs)[0]
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            episode_rewards.append(episode_reward)

        results[name] = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards)
        }

        print(f"   - Recompensa média: {results[name]['mean_reward']:.3f}")

    return results
```

## Troubleshooting

### Problemas Comuns

1. **Dimensões incorretas**: Verificar se os espaços de observação/ação estão corretos
2. **NaN em ações**: Verificar inicialização dos pesos
3. **Convergência lenta**: Ajustar learning rate e arquitetura da rede
4. **Exploração insuficiente**: Aumentar ent_coef ou usar políticas estocásticas

### Debugging

```python
# Verificar forward pass
obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
action = policy.forward(obs_tensor)
print(f"Action shape: {action.shape}")
print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

# Verificar comunicação
if communication is not None:
    comm_tensor = torch.tensor(communication, dtype=torch.float32).unsqueeze(0)
    combined = torch.cat([obs_tensor, comm_tensor], dim=-1)
    print(f"Combined shape: {combined.shape}")
```

## Extensões Futuras

### Políticas Planejadas

1. **HierarchicalPolicy**: Política hierárquica com diferentes níveis
2. **AttentionPolicy**: Política baseada em mecanismos de atenção
3. **GraphPolicy**: Política baseada em grafos para comunicação
4. **MetaPolicy**: Política que aprende a aprender

### Melhorias

1. **Dynamic Communication**: Comunicação adaptativa baseada no contexto
2. **Multi-Objective Policies**: Políticas para múltiplos objetivos
3. **Safe Policies**: Políticas com garantias de segurança
4. **Transfer Learning**: Reutilização de políticas entre datasets

## Referências

- [Stable Baselines3 Policies](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- [Multi-Agent Policy Gradient](https://arxiv.org/abs/1706.02275)
- [Communication in MARL](https://arxiv.org/abs/1912.06069)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)