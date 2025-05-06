# SOC Classification Utils

Standard Occupational Classification (SOC) Utilities, initially developed for Survey Assist API and complements the SOC Classification Library code.

## Overview

SOC classification utilities used in the classification of occupations.  This repository contains core code used by the SOC Classification Library.

## Features

- Embeddings.  Functionality for embedding SOC hierarchy data, managing vector stores,
and performing similarity searches
- Data Access. Functions to load CSV data files related to SOC.

## Prerequisites

Ensure you have the following installed on your local machine:

- [ ] Python 3.12 (Recommended: use `pyenv` to manage versions)
- [ ] `poetry` (for dependency management)
- [ ] Colima (if running locally with containers)
- [ ] Terraform (for infrastructure management)
- [ ] Google Cloud SDK (`gcloud`) with appropriate permissions

### Local Development Setup

The Makefile defines a set of commonly used commands and workflows.  Where possible use the files defined in the Makefile.

#### Clone the repository

```bash
git clone https://github.com/ONSdigital/soc-classification-utils.git
cd soc-classification-utils
```

#### Install Dependencies

```bash
poetry install
```

#### Add Git Hooks

Git hooks can be used to check code before commit. To install run:

```bash
pre-commit install
```

### Run Locally

${\small\color{red}\text{TODO}}$

### Structure

[docs](docs) - documentation as code using mkdocs

[scripts](scripts) - location of any supporting scripts (e.g data cleansing etc)

${\small\color{red}\text{TODO}}$

[src/occupational_classification_utils/data](src/occupational_classification_utils/data) - example data and SOC classification data used for embeddings

[src/occupational_classification_utils/embed](src/occupational_classification_utils/embed) - ChromaDB vector store and embedding code, includes an example use of the store.

[src/occupational_classification_utils/models](src/occupational_classification_utils/models) - common data structures that need to be shared

[src/occupational_classification_utils/utils](src/occupational_classification_utils/utils) - common utility functions such as xls file read for embeddings.

[tests](tests) - PyTest unit testing for code base, aim is for 80% coverage.

### GCP Setup

${\small\color{red}\text{TODO}}$

### Code Quality

Code quality and static analysis will be enforced using isort, black, ruff, mypy and pylint. Security checking will be enhanced by running bandit.

To check the code quality, but only report any errors without auto-fix run:

```bash
make check-python-nofix
```

To check the code quality and automatically fix errors where possible run:

```bash
make check-python
```

### Documentation

Documentation is available in the docs folder and can be viewed using mkdocs

```bash
make run-docs
```

### Testing

${\small\color{red}\text{TODO}}$

### Environment Variables

${\small\color{red}\text{TODO}}$
