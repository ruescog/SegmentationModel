Usage
=====

SegmentationManager
-----

When using this library, you will use the method 
``SegmentationManager.create_model(architecture, backbone, weights, dataloader, num_classes, **kwargs)``
where:

- ``architecture``: is a ``string`` which references the name of the architecture. For example, ``unet``.

- ``backbone``: is a ``string`` which references the name of the backbone. For example, ``resnet18``.

- ``weights``: is optional. Is a ``string`` which references the name of the weights of the ``backbone``. It is used in ``segmentation_models_pytorch`` and ``mmsegmentation`` models.
  
- ``dataloader``: is optional. Is a ``DataLoader`` which defines the training set of the model. It is used in ``semtorch``, ``segmentron`` and ``segmentation_models_pytorch`` models.
  
- ``num_classes``: 2 by default. Is an ``int`` that sets the number of classes to predict by the model.
  
- ``**kwargs**``: are other arguments that can help to create the model.

This method will do the basic actions to create a Learner with the selected characteristics, returning it.

SegmentationModel
-----

Este objeto es una encapsulación del modelo creado. Aunque no sea necesario, se pueden obtener sus
metadatos de la siguiente forma:

- ``libreria``: la librería con la que se ha creado el modelo.

- ``modelo``: el modelo en cuestión.

- ``architecture``: la architecture del modelo.

- ``backbone``: el backbone de modelo.

- ``weights``: los weights del backbone.

Una vez creado el objeto, se podrá entrenar el modelo (independientemente de cuál sea) con la función
``SegmentationModel.entrenar()``.