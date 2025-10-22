# Makefile para o projeto Cooperative Demand Response with MARL

.PHONY: help install test lint format clean train evaluate

help:
	@echo "Comandos disponíveis:"
	@echo "  install    - Instala as dependências do projeto"
	@echo "  test       - Executa os testes"
	@echo "  lint       - Executa análise de código"
	@echo "  format     - Formata o código"
	@echo "  clean      - Limpa arquivos temporários"
	@echo "  train      - Executa o treinamento"
	@echo "  evaluate   - Executa a avaliação"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	flake8 src/ --max-line-length=88
	black --check src/

format:
	black src/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py