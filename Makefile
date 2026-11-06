#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mibici-mlops
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Crea el entorno virtual en .venv (activar con: source .venv/bin/activate)
.PHONY: create_environment
create_environment:
	python$(PYTHON_VERSION) -m venv .venv
	@echo ">>> Entorno creado. Activalo con: source .venv/bin/activate"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Corre la Fase 1 completa (limpieza, features, EDA y versiones en MLflow)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.pipeline

## Ajusta, evalua y registra los modelos en MLflow (Fase 2). Requiere: make data
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mibici.modeling.train

## Genera el pronostico con el modelo champion del Model Registry
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m mibici.modeling.predict

## Abre la interfaz de MLflow: experimentos, comparacion de runs y Model Registry
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
