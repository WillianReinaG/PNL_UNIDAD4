# PNL — Unidad 4

Repositorio de la **Unidad 4** del curso de procesamiento de lenguaje natural (NLP) con **transformers preentrenados** y el ecosistema [Hugging Face](https://huggingface.co/).

## Contenido

- **Notebook principal:** [MIAA/3. SEMESTRE/2. NLP TRANSFORMES/4. unidad/1_text_classification_entrega4.ipynb](MIAA/3.%20SEMESTRE/2.%20NLP%20TRANSFORMES/4.%20unidad/1_text_classification_entrega4.ipynb)  
  Clasificación de texto (sentimiento) sobre textos turísticos en español: carga del dataset, tokenización, comparación de enfoques (featurizer + cabezas y *fine tuning* completo) con `transformers`, `datasets` y `Trainer`.

La estructura de carpetas replica la del proyecto **MIAA** en tu máquina local.

## Requisitos

- Python **3.10** o superior (3.12 suele funcionar bien).
- Espacio en disco y, si entrenas en local, **GPU NVIDIA** recomendable (en CPU el entrenamiento será mucho más lento).

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/WillianReinaG/PNL_UNIDAD4.git
   cd PNL_UNIDAD4
   ```

2. Crea y activa un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Instala dependencias:

   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

   Si usas **GPU**, conviene instalar primero el paquete `torch` acorde a tu CUDA desde la [guía oficial de PyTorch](https://pytorch.org/get-started/locally/) y después ejecutar `pip install -r requirements.txt` para el resto.

4. Abre el notebook:

   ```bash
   jupyter notebook
   ```

   O ábrelo desde **VS Code / Cursor** con la extensión de Jupyter.

## Referencias rápidas (del notebook)

- Dataset: [analisis-sentimientos-textos-turisitcos-mx-polaridad](https://huggingface.co/datasets/alexcom/analisis-sentimientos-textos-turisitcos-mx-polaridad)
- Modelo base: [bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## Autor

Willian Reina — repositorio: [PNL_UNIDAD4](https://github.com/WillianReinaG/PNL_UNIDAD4).
