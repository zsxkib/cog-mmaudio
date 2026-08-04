<div align="center">
<p align="center">
  <h2>MMAudio</h2>
  <a href="https://arxiv.org/abs/2412.15322">Artículo</a> | <a href="https://hkchengrex.github.io/MMAudio">Sitio web</a> | <a href="https://huggingface.co/hkchengrex/MMAudio/tree/main">Modelos</a> | <a href="https://huggingface.co/spaces/hkchengrex/MMAudio">Demostración en Hugging Face</a> | <a href="https://colab.research.google.com/drive/1TAaXCY2-kPk4xE4PwKB3EqFbSnkUuzZ8?usp=sharing">Demostración en Colab</a> | <a href="https://replicate.com/zsxkib/mmaudio">Demostración en Replicate</a>
</p>
</div>

## [Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis](https://hkchengrex.github.io/MMAudio)

[Ho Kei Cheng](https://hkchengrex.github.io/), [Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ), [Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ), [Alexander Schwing](https://www.alexander-schwing.de/), [Yuki Mitsufuji](https://www.yukimitsufuji.com/)

Universidad de Illinois en Urbana-Champaign, Sony AI y Sony Group Corporation

CVPR 2025

## Novedad

MMAudio genera audio sincronizado a partir de entradas de vídeo y/o texto.
Nuestra principal innovación es el entrenamiento conjunto multimodal, que permite entrenar el modelo con una amplia variedad de conjuntos de datos audiovisuales y audio-texto.
Además, un módulo de sincronización alinea el audio generado con los fotogramas del vídeo.

## Resultados

(Todo el audio proviene de nuestro algoritmo MMAudio)

Vídeos de Sora:

https://github.com/user-attachments/assets/82afd192-0cee-48a1-86ca-bd39b8c8f330

Vídeos de Veo 2:

https://github.com/user-attachments/assets/8a11419e-fee2-46e0-9e67-dfb03c48d00e

Vídeos de MovieGen/Hunyuan Video/VGGSound:

https://github.com/user-attachments/assets/29230d4e-21c1-4cf8-a221-c28f2af6d0ca

Para ver más resultados, visita https://hkchengrex.com/MMAudio/video_main.html.


## Instalación

Solo hemos probado este proyecto en Ubuntu.

### Requisitos previos

Recomendamos utilizar un entorno [miniforge](https://github.com/conda-forge/miniforge).

- Python 3.9 o superior
- PyTorch **2.5.1+** y las versiones correspondientes de torchvision y torchaudio (selecciona tu versión de CUDA en https://pytorch.org/, se recomienda instalar con pip)
<!-- - ffmpeg<7 ([esto es requerido por torchaudio](https://pytorch.org/audio/master/installation.html#optional-dependencies), puedes instalarlo en un entorno miniforge con `conda install -c conda-forge 'ffmpeg<7'`) -->

**1. Instala los requisitos previos si aún no están instalados:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

(O cualquier otra versión de CUDA compatible con tus GPU/controladores)

<!-- ```
conda install -c conda-forge 'ffmpeg<7'
```
(Opcional, si usas miniforge y no tienes ya ffmpeg instalado) -->

**2. Clona nuestro repositorio:**

```bash
git clone https://github.com/hkchengrex/MMAudio.git
```

**3. Instala con pip (instala PyTorch primero antes de intentar esto!):**

```bash
cd MMAudio
pip install -e .
```

(Si recibes el error "File 'setup.py' not found", actualiza tu pip con `pip install --upgrade pip`)


**Modelos preentrenados:**

Los modelos se descargarán automáticamente cuando ejecutes el script de demostración. Las sumas de verificación MD5 están disponibles en `mmaudio/utils/download_utils.py`.
Los modelos también están disponibles en https://huggingface.co/hkchengrex/MMAudio/tree/main
Para más detalles, consulta [MODELS.md](docs/MODELS.md).

## Demostración

Por defecto, estos scripts utilizan el modelo `large_44k_v2`.
En nuestros experimentos, la inferencia solo requiere alrededor de 6 GB de memoria GPU (en modo de 16 bits), lo que debería ser suficiente para la mayoría de las GPU modernas.

### Interfaz de línea de comandos

Con `demo.py`

```bash
python demo.py --duration=8 --video=<ruta al video> --prompt "tu frase"
```

La salida (audio en formato .flac y vídeo en formato .mp4) se guardará en `./output`.
Consulta el archivo para más opciones.
Para realizar síntesis de texto a audio, simplemente omite la opción `--video`.
La duración predeterminada de salida (y entrenamiento) es de 8 segundos. Las duraciones más largas o más cortas también pueden funcionar, pero una gran desviación de la duración de entrenamiento puede resultar en una calidad inferior.

### Interfaz Gradio

Soporta síntesis de vídeo a audio y texto a audio.
También puedes probar la síntesis experimental de imagen a audio, que duplica la imagen de entrada en un vídeo para su procesamiento. Esto podría ser interesante para algunos, pero no es para lo que MMAudio ha sido entrenado.
Si es necesario, utiliza [reenvío de puerto](https://unix.stackexchange.com/questions/115897/whats-ssh-port-forwarding-and-whats-the-difference-between-ssh-local-and-remot) (por ejemplo, `ssh -L 7860:localhost:7860 server`). El puerto predeterminado es `7860`, que puedes especificar con `--port`.

```bash
python gradio_demo.py
```

### Preguntas frecuentes

1. Procesamiento de vídeo
    - El procesamiento de vídeos de mayor resolución toma más tiempo debido a la codificación y decodificación (lo que puede tomar >95% del tiempo de procesamiento), pero no mejora la calidad de los resultados.
    - El codificador CLIP redimensiona los fotogramas de entrada a 384×384 píxeles.
    - Synchformer redimensiona el borde más corto a 224 píxeles y aplica un recorte central, enfocándose solo en el cuadrado central de cada fotogramma.
2. Frecuencias de fotogramas
    - El modelo CLIP opera a 8 FPS, mientras que Synchformer funciona a 25 FPS.
    - La conversión de frecuencia de fotogramas ocurre en tiempo real a través del lector de vídeo.
    - Para vídeos de entrada con una frecuencia de fotogramas inferior a 25 FPS, los fotogramas se duplicarán para coincidir con la frecuencia requerida.
3. Casos de fallo
Como con la mayoría de los modelos de este tipo, pueden ocurrir fallos, y las razones no siempre están claras. A continuación se presentan algunos modos de fallo conocidos. Si detectas un modo de fallo o crees que hay un error, no dudes en abrir una issue en el repositorio.
4. Variaciones de rendimiento
Observamos que pueden haber sutiles variaciones de rendimiento en diferentes entornos de hardware y software. Algunas de las razones incluyen el uso o no de `torch.compile`, la biblioteca/backend del lector de vídeo, la precisión de inferencia, los tamaños de lote, las semillas aleatorias, etc. (Lo hará) proporcionaremos resultados precomputados en un benchmark estándar para referencia. Los resultados obtenidos de este código base deberían ser similares pero podrían no ser exactamente los mismos.

### Limitaciones conocidas

1. El modelo a veces genera sonidos similares a la voz humana pero ininteligibles.
2. El modelo a veces genera música de fondo (sin entrenamiento explícito, no sería de alta calidad).
3. El modelo tiene dificultades con conceptos no familiares, por ejemplo, puede generar "disparos de arma" pero no "disparos de RPG".

Creemos que todas estas tres limitaciones pueden ser abordadas con más datos de entrenamiento de alta calidad.

## Entrenamiento

Consulta [TRAINING.md](docs/TRAINING.md).

## Evaluación

Consulta [EVAL.md](docs/EVAL.md).

## Conjuntos de datos de entrenamiento

MMAudio fue entrenado en varios conjuntos de datos, incluyendo [AudioSet](https://research.google.com/audioset/), [Freesound](https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md), [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), [AudioCaps](https://audiocaps.github.io/) y [WavCaps](https://github.com/XinhaoMei/WavCaps). Estos conjuntos de datos están sujetos a licencias específicas, que pueden ser consultadas en sus respectivos sitios web. No garantizamos que los modelos preentrenados sean adecuados para uso comercial. Por favor, úsalos bajo tu propia responsabilidad.

## Registro de actualizaciones

- 2025-03-09: Subimos los archivos tsv corregidos. Consulta [TRAINING.md](docs/TRAINING.md).
- 2025-02-27: Desactivamos el GradScaler por defecto para mejorar la estabilidad del entrenamiento. Consulta #49.
- 2024-12-23: Agregamos scripts de entrenamiento y evaluación por lotes.
- 2024-12-14: Eliminamos el requisito de `ffmpeg<7` para las demostraciones reemplazando `torio.io.StreamingMediaDecoder` con `pyav` para leer fotogramas. Los fotogramas leídos también se almacenan en caché, por lo que no leemos los mismos fotogramas nuevamente durante la reconstrucción. Esto debería acelerar las cosas y hacer que la instalación sea menos problemática.
- 2024-12-13: Mejoramos el procesamiento en bucle para la extracción de características CLIP/Sync introduciendo un multiplicador de tamaño de lote. Podemos utilizar aproximadamente 40x el tamaño de lote para CLIP/Sync sin usar más memoria, acelerando así el procesamiento. Eliminamos el codificador VAE durante la inferencia: no lo necesitamos.
- 2024-12-11: Reemplazamos `torio.io.StreamingMediaDecoder` con `pyav` para leer la frecuencia de fotogramas al reconstruir el vídeo de entrada. `torio.io.StreamingMediaDecoder` no funciona de manera confiable en el entorno ZeroGPU de HuggingFace, y sospecho que podría no funcionar en algunos otros entornos también.

## Cita

```bibtex
@inproceedings{cheng2025taming,
  title={Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis},
  author={Cheng, Ho Kei and Ishii, Masato and Hayakawa, Akio and Shibuya, Takashi and Schwing, Alexander and Mitsufuji, Yuki},
  booktitle={CVPR},
  year={2025}
}
```

## Repositorios relevantes

- [av-benchmark](https://github.com/hkchengrex/av-benchmark) para resultados de referencia.

## Descargo de responsabilidad

No tenemos afiliación con ni tenemos conocimiento de la parte detrás del dominio "mmaudio.net".

## Agradecimiento

Muchos gracias a:
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) por el modelo BigVGAN preentrenado de 16kHz y la arquitectura VAE
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [Synchformer](https://github.com/v-iashin/Synchformer) 
- [EDM2](https://github.com/NVlabs/edm2) por la arquitectura de red VAE con preservación de magnitud
