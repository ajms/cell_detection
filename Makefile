define find.functions
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[31m%20s\033[0m\t%s\n", $$1, $$2}'
endef

.PHONY: help
help:  ## help
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)

.PHONY: prj-init
prj-init: ## project initialisation
	poetry install
	poetry run pre-commit install
	poetry run pre-commit run --all check-added-large-files
	poetry run pre-commit run --all check-merge-conflict
	poetry run pre-commit run --all check-yaml
	poetry run pre-commit run --all detect-private-key
	poetry run pre-commit run --all end-of-file-fixer
	poetry run pre-commit run --all trailing-whitespace
	poetry run pre-commit run --all isort
	poetry run pre-commit run --all black
	poetry run pre-commit run --all flake8
	poetry run pytest

.PHONY: test
test:  ## run pytest
	pytest . -p no:logging -p no:warnings

.PHONY: lint
lint:  ## run linting
	isort src
	black src
	flake8 src

.PHONY: pre-commit
pre-commit:  ## run all pre-commit checks
	pre-commit --run all

.PHONY: aim
aim:  ## run aim UI
	cd data/cell-detection/aim; aim up
