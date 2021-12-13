.. _instalación: Instalación
=====

Para instalar ``segmentationmodel``, se debe ejecutar el siguiente comando (usando pip):

.. code-block:: console

   $ pip install segmentationmodel

Este comando instalará todos los paquetes necesarios para usar la librería excepto los relacionados con ``mmsegmentation``.

Si vas a hacer uso de la librería ``mmsegmentation``, deberás instalar manualmente sus paquetes:

.. code-block:: console

   $ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

Si además deseas hacer uso de CUDA para aumentar la velocidad de entrenamiento de los modelos, deberás actualizar las versiones de ``torch`` y ``torchvision`` para que puedan hacer uso de CUDA. Si es así, instala:

.. code-block:: console

   $ pip install 'torch==1.6.0+cu101' -f https://download.pytorch.org/whl/torch_stable.html

.. code-block:: console

   $ pip install 'torchvision==0.7.0+cu101' -f https://download.pytorch.org/whl/torch_stable.html

.. note::

   ``mmsegmentation`` es un repositorio que está en desarrollo actualmente, por lo que las versiones de las librerías ``mmcv-full``, ``torch`` y ``torchvision`` podrían necesitar una versión más actualizada.