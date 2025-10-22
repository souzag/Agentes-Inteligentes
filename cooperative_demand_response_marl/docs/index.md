# Documentação do Projeto

## Visão Geral
Este projeto implementa um sistema de Resposta Cooperativa à Demanda (Demand Response) utilizando Aprendizado por Reforço Multi-Agente (Multi-Agent Reinforcement Learning - MARL).

## Objetivos
- Desenvolver agentes inteligentes que cooperem para otimizar o consumo de energia
- Reduzir picos de demanda em redes elétricas
- Melhorar a eficiência energética através de cooperação entre agentes

## Arquitetura
O sistema é composto por:
- **Agentes**: Entidades inteligentes que tomam decisões de consumo
- **Ambiente**: Simulação do sistema de energia
- **Algoritmos**: Implementações de MARL para treinamento
- **Utilitários**: Ferramentas auxiliares

## Instalação
```bash
pip install -r requirements.txt
```

## Uso Básico
```bash
# Treinar modelo
python scripts/train.py

# Avaliar modelo
python scripts/evaluate.py
```

## Estrutura
- `src/`: Código fonte principal
- `data/`: Dados do projeto
- `models/`: Modelos treinados
- `results/`: Resultados e análises
- `scripts/`: Scripts auxiliares
- `tests/`: Testes unitários e de integração