# Proyecto mi bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

El dataset abierto de MiBici, sistema de bicicletas públicas del Área Metropolitana de Guadalajara, proporciona información histórica sobre el uso del servicio desde diciembre de 2014 hasta septiembre de 2026, con más de 39 millones de viajes registrados. Los datos están disponibles en archivos CSV organizados por mes y año, permitiendo analizar patrones de movilidad urbana, estaciones de origen y destino, horarios de mayor demanda y tendencias temporales. Este conjunto de datos representa una excelente oportunidad para desarrollar un proyecto de Machine Learning (ML) utilizando la metodología MLOps, donde podrán implementar modelos de predicción de demanda, identificar patrones de movilidad, detectar anomalías y estimar necesidades de redistribución de bicicletas etc. El proyecto permitirá trabajar con datos reales y aplicar prácticas de ingeniería de datos, versionamiento de datasets, entrenamiento y evaluación de modelos, automatización de pipelines, despliegue y monitoreo del desempeño de modelos de ML a lo largo del tiempo.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

