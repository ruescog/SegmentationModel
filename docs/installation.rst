Installation
=====

To install ``segmentationmodel``, the following command must be executed (using pip):

.. code-block:: console

   $ pip install segmentationmodel

This command will install all the packages needed to use the library except those related to
``mmsegmentation``.

If you are going to make use of the ``mmsegmentation`` library, you will need to manually install its
packages:

.. code-block:: console

   $ pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

If you also want to use CUDA to increase the training speed of the models (recomended), you need to
update the ``torch`` and ``torchvision`` versions to support CUDA, you need to install:

.. code-block:: console

   $ pip install 'torch==1.6.0+cu101' -f https://download.pytorch.org/whl/torch_stable.html

.. code-block:: console

   $ pip install 'torchvision==0.7.0+cu101' -f https://download.pytorch.org/whl/torch_stable.html

.. note::

   ``mmsegmentation`` is a repository that is under development. Its libraries versions may change in
   the future.