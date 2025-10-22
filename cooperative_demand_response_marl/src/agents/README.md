# Agentes MARL para Demand Response Cooperativo

## Visão Geral

Este diretório contém a implementação de agentes MARL (Multi-Agent Reinforcement Learning) para o sistema de demand response cooperativo baseado no ambiente CityLearn. Os agentes são projetados para aprender políticas de controle de HVAC e armazenamento de energia que otimizam tanto objetivos individuais quanto coletivos.

## Arquitetura dos Agentes

### Tipos de Agentes

#### 1. IndependentAgent
Agente que aprende independentemente, sem considerar outros agentes.

```python
class IndependentAgent:
    """Agente que aprende política individual sem cooperação"""
    - Política: Rede neural individual
    - Recompensa: Baseada apenas no próprio estado
    - Comunicação: Nenhuma
```

#### 2. CooperativeAgent
Agente que considera informações de outros agentes para tomada de decisão.

```python
class CooperativeAgent:
    """Agente que coopera com outros agentes"""
    - Política: Rede neural com embeddings de outros agentes
    - Recompensa: Combinação local + global + cooperação
    - Comunicação: Estado global compartilhado
```

#### 3. CentralizedAgent
Agente centralizado que controla todos os prédios.

```python
class CentralizedAgent:
    """Agente centralizado para controle global"""
    - Política: Rede neural centralizada
    - Recompensa: Baseada no estado global
    - Comunicação: Controle direto de todos os prédios
```

## Implementações

### Arquivos

1. **`base_agent.py`**: Classe base para todos os agentes
2. **`independent_agent.py`**: Implementação do agente independente
3. **`cooperative_agent.py`**: Implementação do agente cooperativo
4. **`centralized_agent.py`**: Implementação do agente centralizado
5. **`agent_factory.py`**: Factory para criar agentes
6. **`policies.py`**: Políticas customizadas para SB3

### BaseAgent

```python
class BaseAgent:
    """Classe base para agentes MARL"""

    def __init__(self, env, agent_id, config):
        self.env = env
        self.agent_id = agent_id
        self.config = config
        self.policy = None
        self.training_history = []

    @abstractmethod
    def select_action(self, observation):
        """Seleciona ação baseada na observação"""
        pass

    @abstractmethod
    def update_policy(self, experience):
        """Atualiza política baseada na experiência"""
        pass

    def get_info(self):
        """Retorna informações do agente"""
        return {
            "agent_id": self.agent_id,
            "policy_type": type(self.policy).__name__,
            "training_steps": len(self.training_history)
        }
```

## Políticas para Stable Baselines3

### Custom Policies

#### 1. MultiAgentPolicy
Política que considera múltiplos agentes.

```python
class MultiAgentPolicy(BasePolicy):
    """Política para múltiplos agentes com comunicação"""

    def __init__(self, observation_space, action_space, num_agents, communication_dim=0):
        super().__init__(observation_space, action_space)
        self.num_agents = num_agents
        self.communication_dim = communication_dim

        # Rede neural com camadas para comunicação
        self.shared_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Camada de comunicação
        if communication_dim > 0:
            self.comm_net = nn.Linear(communication_dim, 64)

        # Cabeça da política
        self.action_net = nn.Linear(128 + (64 if communication_dim > 0 else 0), action_space.shape[0])
```

#### 2. CooperativePolicy
Política especificamente para cooperação.

```python
class CooperativePolicy(BasePolicy):
    """Política otimizada para cooperação"""

    def forward(self, obs, communication=None):
        # Processar observação
        features = self.shared_net(obs)

        # Processar comunicação se disponível
        if communication is not None:
            comm_features = self.comm_net(communication)
            combined = torch.cat([features, comm_features], dim=-1)
        else:
            combined = features

        # Gerar ação
        action = self.action_net(combined)
        return action
```

## Algoritmos Suportados

### 1. PPO (Proximal Policy Optimization)
```python
# Configuração para agente cooperativo
ppo_config = {
    "policy": "MultiAgentPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01
}
```

### 2. SAC (Soft Actor-Critic)
```python
# Configuração para ações contínuas
sac_config = {
    "policy": "MultiAgentPolicy",
    "learning_rate": 3e-4,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1
}
```

### 3. MADDPG (Multi-Agent DDPG)
```python
# Configuração para múltiplos agentes
maddpg_config = {
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "gamma": 0.95,
    "tau": 0.01,
    "batch_size": 256,
    "memory_size": 1000000
}
```

## Sistema de Comunicação

### Integração com Comunicação

```python
class CommunicationAwareAgent(BaseAgent):
    """Agente que usa comunicação para tomada de decisão"""

    def __init__(self, env, agent_id, communication_protocol):
        super().__init__(env, agent_id)
        self.comm_protocol = communication_protocol

    def select_action(self, observation, messages=None):
        """Seleciona ação considerando mensagens de outros agentes"""
        # Processar observação local
        action = self.policy.predict(observation)

        # Considerar mensagens se disponíveis
        if messages:
            action = self._incorporate_messages(action, messages)

        return action

    def _incorporate_messages(self, action, messages):
        """Incorpora informações de mensagens na ação"""
        # Implementação específica para cada tipo de agente
        pass
```

## Configuração

### Arquivo de Configuração

```yaml
# config/agents.yaml
agents:
  type: "cooperative"  # "independent", "cooperative", "centralized"
  num_agents: 5

  policy:
    name: "MultiAgentPolicy"
    shared_parameters: true
    communication_dim: 32

  training:
    algorithm: "PPO"
    total_timesteps: 1000000
    eval_freq: 10000
    save_freq: 50000

  communication:
    enabled: true
    protocol: "full"  # "full", "neighborhood", "centralized"
    message_dim: 16

  reward:
    local_weight: 0.4
    global_weight: 0.4
    cooperation_weight: 0.2
```

## Exemplos de Uso

### Agente Independente

```python
from src.agents import IndependentAgent
from src.environment import make_citylearn_vec_env

# Criar ambiente
env = make_citylearn_vec_env("citylearn_challenge_2022_phase_1")

# Criar agente independente
agent = IndependentAgent(env, agent_id=0, config={})

# Treinamento
for episode in range(1000):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        agent.update_policy((obs, action, reward, obs, done))
```

### Agente Cooperativo

```python
from src.agents import CooperativeAgent
from src.communication import FullCommunication

# Criar protocolo de comunicação
comm_protocol = FullCommunication(num_agents=5)

# Criar agente cooperativo
agent = CooperativeAgent(env, agent_id=0, communication_protocol=comm_protocol)

# Treinamento com comunicação
for episode in range(1000):
    obs, info = env.reset()
    done = False

    while not done:
        # Receber mensagens de outros agentes
        messages = comm_protocol.receive_messages(agent.agent_id)

        # Selecionar ação considerando comunicação
        action = agent.select_action(obs, messages)

        # Enviar estado para outros agentes
        comm_protocol.send_message(agent.agent_id, "all", obs)

        obs, reward, done, info = env.step(action)
        agent.update_policy((obs, action, reward, obs, done))
```

## Testes e Validação

### Testes Unitários

```python
def test_agent_creation():
    """Testa criação de agentes"""
    env = make_citylearn_vec_env()
    agent = IndependentAgent(env, 0, {})
    assert agent.agent_id == 0
    assert agent.policy is not None

def test_action_selection():
    """Testa seleção de ações"""
    env = make_citylearn_vec_env()
    agent = IndependentAgent(env, 0, {})

    obs, info = env.reset()
    action = agent.select_action(obs)

    assert env.action_space.contains(action)
```

### Testes de Integração

```python
def test_multi_agent_training():
    """Testa treinamento multi-agente"""
    env = make_citylearn_vec_env()
    agents = [IndependentAgent(env, i, {}) for i in range(5)]

    # Simular treinamento
    for episode in range(10):
        obs, info = env.reset()
        episode_reward = 0

        while True:
            actions = [agent.select_action(obs) for agent in agents]
            obs, rewards, done, info = env.step(actions)

            # Atualizar agentes
            for i, agent in enumerate(agents):
                agent.update_policy((obs, actions[i], rewards[i], obs, done))

            episode_reward += sum(rewards)

            if done:
                break

        print(f"Episode {episode}: Reward = {episode_reward}")
```

## Performance e Otimização

### Métricas de Performance

```python
def evaluate_agent_performance(agent, env, num_episodes=10):
    """Avalia performance do agente"""
    total_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "min_reward": np.min(total_rewards),
        "max_reward": np.max(total_rewards)
    }
```

### Otimizações

1. **Parameter Sharing**: Compartilhar parâmetros entre agentes similares
2. **Communication Compression**: Comprimir mensagens de comunicação
3. **Batch Processing**: Processar múltiplos agentes em paralelo
4. **Experience Replay**: Reutilizar experiências de treinamento

## Extensões Futuras

### Funcionalidades Planejadas

1. **Hierarchical Agents**: Agentes com diferentes níveis de decisão
2. **Transfer Learning**: Reutilização de políticas entre datasets
3. **Meta-Learning**: Adaptação rápida a novos prédios
4. **Safe RL**: Garantias de segurança e conforto

### Integrações

- **Ray RLlib**: Algoritmos distribuídos
- **Weights & Biases**: Tracking de experimentos
- **TensorBoard**: Visualização de treinamento
- **Custom Environments**: Suporte a novos ambientes

## Referências

- [Multi-Agent PPO](https://arxiv.org/abs/2103.01955)
- [MADDPG](https://arxiv.org/abs/1706.02275)
- [Communication in MARL](https://arxiv.org/abs/1912.06069)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## Contribuição

Para contribuir:

1. Implementar novos tipos de agentes
2. Adicionar políticas customizadas
3. Otimizar performance
4. Expandir documentação