# PNL — Unidad 4: clasificación de texto con transformers preentrenados

Repositorio de la **Unidad 4** del curso de NLP (MIAA): se construye un **clasificador de sentimientos / polaridad** para **textos turísticos en español (México)** usando un **BERT multilingüe** ya entrenado en Hugging Face, sin entrenar un transformer desde cero. El trabajo práctico vive en el notebook; aquí se resume **qué debes ejecutar, entender y comparar**.

**Notebook:** [MIAA/3. SEMESTRE/2. NLP TRANSFORMES/4. unidad/1_text_classification_entrega4.ipynb](MIAA/3.%20SEMESTRE/2.%20NLP%20TRANSFORMES/4.%20unidad/1_text_classification_entrega4.ipynb)

La estructura de carpetas bajo `MIAA/` replica la de tu proyecto local.

---

## Objetivo del trabajo

- **Tarea:** clasificación multiclase de sentimiento sobre reseñas turísticas.
- **Idea central:** reutilizar un checkpoint **preentrenado** (`nlptown/bert-base-multilingual-uncased-sentiment`) y **adaptarlo** a las etiquetas del dataset mexicano mediante cabezas de clasificación y, al final, **fine tuning** del modelo completo.
- **Meta pedagógica:** recorrer todo el pipeline moderno con [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index), [Datasets](https://huggingface.co/docs/datasets/en/index) y [Evaluate](https://huggingface.co/docs/evaluate/en/index): datos → tokenizer → modelo → entrenamiento → evaluación → interpretación cualitativa.

---

## Dataset y etiquetas

- **Origen:** [alexcom/analisis-sentimientos-textos-turisitcos-mx-polaridad](https://huggingface.co/datasets/alexcom/analisis-sentimientos-textos-turisitcos-mx-polaridad) (split `train` en el notebook).
- **Campos relevantes:** texto (`text`) y etiqueta categórica (`label`).
- **Preparación en el notebook:** se construyen mapas `category2id` / `id2category` a partir de las etiquetas únicas y se añade `label_id`; los ejemplos se **barajan** (`sample(frac=1)`) antes de particionar.

### Análisis exploratorio (lo que debes hacer antes de entrenar)

1. **Distribución de clases:** conteos y gráfico de barras para ver **desbalance** entre categorías.
2. **Longitud de textos:** palabras por texto (boxplot por categoría) y **medianas** por clase para justificar la longitud máxima al tokenizar.
3. **Textos largos:** con umbral de **100 palabras**, conteo por categoría de cuántos textos superan ese umbral (contexto para truncamiento vs. pérdida de información).

---

## Modelo y tokenizer

- **Checkpoint:** [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) (BERT multilingüe orientado a sentimiento).
- **Regla importante:** el **tokenizer** debe ser el del mismo checkpoint (`AutoTokenizer.from_pretrained`), para que vocabulario, tokens especiales y reglas de subpalabras coincidan con el preentrenamiento.
- **Pruebas en el notebook:** tokenización de ejemplo, `vocab_size`, `model_max_length`, `model_input_names` (qué tensores espera el modelo: p. ej. `input_ids`, `attention_mask`, etc.).

---

## Partición train / validación / prueba

Con `datasets`:

1. **80%** entrenamiento, **20%** restante.
2. Ese 20% se divide a la mitad: **~10% validación** y **~10% prueba**.

El conjunto de **prueba** no debe usarse para ajustar hiperparámetros; sirve para la **estimación final** de generalización.

---

## Preprocesamiento para el modelo

- **Tokenización:** `max_length=512`, `truncation=True`, `padding='max_length'` sobre todos los splits (mismas dimensiones por lote, consumo de memoria predecible).
- **Etiquetas:** función que convierte el nombre de categoría a entero (`category_names_2_ids`) y se aplica con `.map()` al `DatasetDict` tokenizado.

---

## Tres experimentos a ejecutar y comparar

El notebook entrena **tres variantes** con el mismo `TrainingArguments` y la misma métrica (**accuracy** vía `evaluate`), de modo que la comparación sea coherente.

### 1. Encoder congelado + cabeza lineal (línea base)

- `AutoModelForSequenceClassification` con `num_labels = número de clases del dataset`.
- Se congelan los parámetros de `model.base_model`; solo aprende la **cabeza** por defecto.
- **Entrenamiento:** `Trainer` + `TrainingArguments` (ver hiperparámetros abajo).
- **Evaluación:** métricas en validación por época; luego `evaluate` en **test**.
- **TensorBoard:** logs en `./hf` con `report_to='tensorboard'`; en el notebook se carga con `%tensorboard --logdir hf/runs`.

### 2. Encoder congelado + cabeza más profunda (MLP propia)

- Misma base congelada.
- Se **reemplaza** `model.classifier` por un `nn.Sequential`: capas lineales **768 → 512 → 256 → num_labels**, ReLU, Dropout 0.2, y **LogSoftmax** en la salida.
- Mismo `Trainer` y mismos argumentos que el experimento 1 para comparar el efecto de **más capacidad solo en la cabeza**.

### 3. Fine tuning completo

- Se vuelve a cargar el modelo desde el checkpoint **sin** congelar el encoder: se entrenan **todas** las capas (encoder + clasificador).
- Misma configuración de entrenamiento que en los casos anteriores (en el notebook se reutiliza el mismo `training_args`).
- **Ventaja esperada:** mejores representaciones adaptadas al dominio; **coste:** más tiempo, memoria y riesgo de sobreajuste si el dataset es limitado.

---

## Hiperparámetros principales (según el notebook)

| Parámetro | Valor |
|-----------|--------|
| Épocas | 2 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Batch size | 8 en Colab, **4** en local (`IN_COLAB`) |
| Evaluación / guardado | Por época (`eval_strategy`, `save_strategy`); `load_best_model_at_end=True` |
| Métrica | `accuracy` (argumentos máximos de logits) |
| Salida | `./hf` |

Ajustar épocas o batch size puede ser necesario en máquinas sin GPU.

---

## Uso del modelo después de entrenar

1. **Predicciones** sobre el conjunto de test con `trainer.predict`.
2. **Tabla** texto, etiqueta real, etiqueta predicha (id y nombre).
3. **Análisis de errores:** filas donde `label != prediction_label` para discutir confusiones entre clases, ambigüedad o ruido en las etiquetas.

Esto complementa la accuracy con una lectura **cualitativa** del comportamiento del clasificador.

---

## Conclusiones que el notebook espera que articules

- Los transformers preentrenados aportan **representaciones lingüísticas fuertes** sin coste de preentrenar desde cero.
- El **tokenizer del checkpoint** no es opcional: es parte del contrato de entrada del modelo.
- **Solo cabeza:** rápido y barato en cómputo; buena línea base.
- **Cabeza profunda + encoder fijo:** puede mejorar la frontera de decisión sin pagar aún el fine tuning completo.
- **Fine tuning completo:** suele dar el mejor rendimiento a cambio de más recursos y más cuidado con el sobreajuste.

---

## Entorno: Google Colab vs. máquina local

- En **Colab**, las primeras celdas instalan dependencias (`lightning`, `datasets`, `transformers[torch]`, etc.) y detectan Colab vía `pkg_resources`.
- En **local**, usa el archivo `requirements.txt` de este repo y, si entrenas sin GPU, espera tiempos mucho mayores.

---

## Requisitos e instalación

- Python **3.10+** recomendado.
- **GPU** muy recomendable para los tres entrenamientos secuenciales.

```bash
git clone https://github.com/WillianReinaG/PNL_UNIDAD4.git
cd PNL_UNIDAD4
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -U pip
pip install -r requirements.txt
```

Para **PyTorch con GPU/CUDA**, sigue primero [Get Started (PyTorch)](https://pytorch.org/get-started/locally/) y luego instala el resto con `requirements.txt`.

---

## Referencias bibliográficas y enlaces (del notebook)

- Dataset: [analisis-sentimientos-textos-turisitcos-mx-polaridad](https://huggingface.co/datasets/alexcom/analisis-sentimientos-textos-turisitcos-mx-polaridad)
- Modelo: [bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- [BERT (Devlin et al., arXiv)](http://arxiv.org/abs/1810.04805)
- [Natural Language Processing with Transformers (O'Reilly)](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246)
- Notebook de referencia (Colab badge en la primera celda): [icesi-nlp / Sesion4](https://colab.research.google.com/github/Ohtar10/icesi-nlp/blob/main/Sesion4/1-text-classification-with-hf.ipynb)

---

## Integrantes

- Juan Manuel Hurtado  
- Manuel Alberto González  
- Willian Alberto Reina  

Repositorio del equipo: [PNL_UNIDAD4 en GitHub](https://github.com/WillianReinaG/PNL_UNIDAD4).
