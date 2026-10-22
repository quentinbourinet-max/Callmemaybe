# Variables
PYTHON_VERSION = 3.10
MAIN_SCRIPT = src

# Règle par défaut
all: install lint run

# INSTALL : Installe uv si absent, puis installe les dépendances
install:
	@echo "Installation des dépendances avec uv..."
	uv sync --link-mode=copy

# RUN : Exécute le script principal via l'environnement virtuel de uv
run:
	uv run python -m $(MAIN_SCRIPT)

# DEBUG : Lance le debugger pdb
debug:
	uv run python -m pdb $(MAIN_SCRIPT)

# CLEAN : Nettoyage des caches et fichiers temporaires
clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# LINT : Vérification du code (Flake8 et Mypy avec les flags imposés)
lint:
	uv run flake8 . --exclude .venv
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# LINT-STRICT : Version renforcée du lint
lint-strict:
	uv run flake8 . --exclude .venv
	uv run mypy . --strict