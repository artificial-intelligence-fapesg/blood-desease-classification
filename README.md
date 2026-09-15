\# VitaLabs Sangue

Interface Streamlit de apoio à classificação de exames de sangue em quatro categorias: Saudável, Hipertenso, Leucemia e Diabetes.

## Funcionalidades

- Cadastro do paciente e dos resultados laboratoriais.
- Análise por modelo individual ou votação entre modelos.
- Exibição de probabilidades, métricas e histórico salvo em SQLite.

## Como usar

```bash
pip install -r requirements.txt
streamlit run app_sangue.py
```

## Adicionar modelos

Coloque o arquivo `.h5`, `.hdf5` ou `.keras` em `models_sangue/`, usando exatamente o nome de uma chave de `MODEL_CATALOG` em `blood_detector.py` (por exemplo, `VitaBL98A.h5`). Ao reiniciar a aplicação, o modelo será disponibilizado automaticamente na interface. Opcionalmente, adicione o scaler correspondente como `VitaBL98A.pkl`.

> Ferramenta de apoio à decisão clínica; não substitui a avaliação médica.
