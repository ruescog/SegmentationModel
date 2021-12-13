Uso
=====

A la hora de usar la librería, se utilizará la función
``SegmentationManager.crear_modelo(arquitectura, esqueleto, pesos, dataloader, num_clases, **kwargs)``.
Donde:

- ``arquitectura`` es un ``string`` con el nombre de la arquitectura. Por ejemplo, ``unet``.

- ``esqueleto`` es un ``string`` con el nombre de esqueleto. Por ejemplo, ``resnet18``.

- ``pesos``: es opcional. Es un ``string`` con el nombre que identifique los pesos del ``esqueleto``. Se usa en los modelos de ``segmentation_models_pytorch`` y ``mmsegmentation``.
  
- ``dataloader``: es opcional. Es un ``DataLoader`` que define el conjunto de entrenamiento del modelo. Se usa en los modelos de ``semtorch``, ``segmentron`` y ``segmentation_models_pytorch``
  
- ``num_clases``: por defecto es 2. Es un ``int`` con el número de clases que el modelo debe predecir.
  
- ``**kwargs**``: son otros argumentos que puedan servir para crear el modelo.
  
y esto es todo amigos =).