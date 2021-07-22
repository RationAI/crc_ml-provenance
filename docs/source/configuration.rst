Pipeline Configuration
======================
Pipeline's python runner uses a single JSON file to confugire the pipeline.


* :ref:`Example Configuration <example_cfg>`

* :ref:`Configuration Details<config_details>`

    * ``experiment`` :ref:`Block <experiment>`
        * ``seed``
        * ``experiment_prefix``
        * ``ordered_steps`` Array
        * ``continue`` Block
            * ``eid``
            * ``copy_data``

    * ``data`` :ref:`Block <data>`
        * ``meta``
            * ``level``
            * ``tile_size``
            * ``center_size``
            * ``include_annot_keywords``

        * ``dirs``
            * ``results``
            * ``root``
            * ``rgb``
            * ``label``
            * ``coord_maps``
            * ``dataset``
                * ``name``
                * ``valid_size``
                * ``test_size``
                * ``pd_label_col``
                * ``stratify_col``

    * ``generator``  :ref:`Block <generator>`
        * ``batch_size``
        * ``valid_generator_type``
        * ``steps_per_epoch``
        * ``validation_steps``
        * ``train_augment``
        * ``valid_augment``
        * ``test_augment``
        * ``extractor`` Block
        * ``augmenter`` Block
        * ``sampler``
            * ``sampling_strat`` Block
            * ``sequence_column``


    * ``model``  :ref:`Block <model>`
        * ``class_name``
        * ``keras_application`` Block
        * ``optimizer`` Block
        * ``loss`` Block
        * ``regularizer`` Block
        * ``input_shape`` Array
        * ``output_size``
        * ``train_checkpoint``
        * ``test_checkpoint``
        * ``output_bias``
        * ``trainable``
        * ``dropout``
        * ``sequence_metrics`` Array
        * ``total_metrics`` Array

    * ``step_definitions`` Dict :ref:`Block <stepdefs>`




.. _example_cfg:

Example Configuration
---------------------

.. code-block:: json

    {

        "experiment": {
            "seed": 0,
            "experiment_prefix": "MY-EXPERIMENT",

            "ordered_steps": [
                "exp.train",
                "exp.test",
                "saliency"
            ]
        },

        "data": {
            "dirs": {
                "results": "models",
                "root": "Prostate",
                "rgb": "slides",
                "label": "annotations",
                "coord_maps": "dataset-L1-T512",

                "dataset": {
                    "name": "dataset-L1-T512.h5",
                    "composed_type": true,
                    "valid_size": 0.1,
                    "pd_label_col": "center_tumor_tile"
                }
            },

            "meta": {
                "level": 1,
                "tile_size": 512,
                "center_size": 256
            }
        },

        "generator": {
            "batch_size": 1,
            "valid_generator_type": "linear",
            "steps_per_epoch": 5000,
            "validation_steps": 1000,

            "train_augment": true,
            "valid_augment": false,
            "test_augment": false,

            "extractor": {
                "class_name": "BinaryClassExtractor"
            },

            "augmenter": {
                "class_name": "SlideAugmenter",
                "config": {
                    "horizontal": 0.5,
                    "vertical": 0.5,
                    "brightness": [-64, 64],
                    "hue": [-10, 10],
                    "saturation": [-64, 64],
                    "contrast": [0.7, 1.3]
                }
            },

            "sampler": {
                "sampling_strat": {
                    "center_tumor_tile": [],
                    "slide_name": []
                },
                "sequence_column": "slide_name"
            }
        },

        "model": {
            "class_name": "PretrainedModel",
            "keras_application": {
                "class_name": "VGG16",
                "config": {
                    "pooling": "max"
                }
            },

            "optimizer": {
                "class_name": "Adam",
                "config": {
                    "lr": 5e-6
                }
            },

            "loss": {
                "class_name": "BinaryCrossentropy"
            },

            "regularizer": {
                "class_name": "L2",
                "config": {
                    "l2": 5e-05
                }
            },

            "input_shape": [
                512,
                512,
                3
            ],

            "output_size": 1,
            "train_checkpoint": "models/checkpoints/VGG16-TF2-DATASET-e95b-4e8f-aeea-b87904166a69/final.hdf5",
            "test_checkpoint": "early.hdf5",
            "output_bias": null,
            "trainable": true,
            "dropout": 0.5,

            "sequence_metrics": [
                {"class_name": "BinaryAccuracy"},
                {"class_name": "Precision"},
                {"class_name": "F1"},
                {"class_name": "AUC"}
            ],
            "total_metrics": [
                {"class_name": "Recall"},
                {"class_name": "Specificity"}
            ]

        },

        "step_definitions": {
            "exp.train": {
                "init": {
                    "class_id": "rationai.training.ExperimentRunner",
                    "config": {
                        "experiment_class": "ClassificationExperiment"
                    }
                },
                "exec": {
                    "method": "train",
                    "kwargs": {
                        "fit_kwargs": {
                            "epochs": 1,
                            "workers": 15,
                            "use_multiprocessing": true,
                            "max_queue_size": 50,
                            "callbacks": [
                                {
                                    "class_name": "ModelCheckpoint",
                                    "config": {
                                        "filepath": "final.hdf5",
                                        "verbose": 1,
                                        "save_best_only": false,
                                        "save_weights_only": true,
                                        "period": 1
                                    }
                                },
                                {
                                    "class_name": "ModelCheckpoint",
                                    "config": {
                                        "filepath": "early.hdf5",
                                        "monitor": "val_auc",
                                        "verbose": 1,
                                        "save_best_only": true,
                                        "save_weights_only": true,
                                        "period": 1,
                                        "mode": "max"
                                    }

                                },
                                {
                                    "class_name": "ReduceLROnPlateau",
                                    "config": {
                                        "monitor": "val_loss",
                                        "factor": 0.5,
                                        "min_lr": 1e-07,
                                        "mode": "max",
                                        "patience": 3
                                    }
                                },
                                {
                                    "class_name": "EarlyStopping",
                                    "config": {
                                        "monitor": "val_auc",
                                        "patience": 5,
                                        "mode": "max",
                                        "restore_best_weights": true
                                    }
                                }
                            ]
                        }
                    }
                }
            },

            "exp.test": {
                "exec": {
                    "method": "infer",
                    "kwargs": {
                        "infer_type": "predict_evaluate",
                        "infer_params": {
                            "workers": 10,
                            "max_queue_size": 50,
                            "use_multiprocessing": true
                        }
                    }
                }
            },

            "saliency": {
                "init": {
                    "class_id": "rationai.visual.explain.SaliencyRunner",
                    "config": {
                        "grad_modifier": null
                    }
                },
                "exec": {
                    "method": "run",
                    "kwargs": {
                        "slide_names": ["TP-2019_6785-13-1"]
                    }
                }
            }
        }
    }


.. _config_details:

Configuration Details
---------------------


.. ==== EXPERIMENT=================

.. _experiment:

``experiment`` Block
````````````````````
    The block defines pipeline steps and behaviour.

.. _fix_seed:

``seed``
^^^^^^^^
    Sets the seed for Numpy's ``numpy.random.seed`` and Tensorflow's ``tf.random.set_seed()``.
    If no seed is provided, a new one is generated and logged for reproducibility.

        *Type*: int


``experiment_prefix``
^^^^^^^^^^^^^^^^^^^^^
    Prefixes the experiment ID. Results directory's name is same as the experiment ID.

        *Type:* string



``ordered_steps``
^^^^^^^^^^^^^^^^^
    Declares the pipeline's steps and their order.
    It is a subset of user defined keys, whose steps are defined in :ref:`step_definitions <stepdefs>`.

        *Type:* array

    Example:

        .. code-block:: json

            {
                "ordered_steps": ["test", "occlusions"]
            }

    .. _context:

    .. note::
        A set of steps with the same prefix delimited by a dot are part of one context (i.e., are considered to be a single instance).
        Pipeline's *step executor* keeps the step's object instantiated for future usage, and releases the resources after the last usage.
        This allows to run multiple methods (or a method multiple times) for a single step instance.
        Example:

        .. code-block:: json

            {
                "ordered_steps": ["exp.train", "pick_ckpt", "exp.test"]
            }


``continue`` Block (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Block used to perform a follow-up of a previous experiment. It will copy and access the existing results.

    Example:

    .. code-block:: json

        {
            "continue": {
                "eid": "MY_PREVIOUS_TRAINING_EXPERIMENT",
                "copy_data": true
            }
        }


    String ``eid`` specifies the ID of an experiment to continue.

    If ``copy_data`` is set to ``True``, the results of the previous experiment are copied to the directory of the current experiment, where they can be used.

    .. warning::
        Should not be used at *Metacentrum*.

        Only ``copy_data=True`` is supported at the moment.



.. ==== DATA ======================

.. _data:

``data`` Block
``````````````

Defines information about data and data structure on disk.

``meta`` Block
^^^^^^^^^^^^^^
Contains meta data on how the tiles were sampled from a WSI. The required fields depend on the type of methods being computed.

    .. list-table::
        :widths: 15 15 60
        :header-rows: 1

        * - Field
          - Type
          - Details
        * - ``level``
          - integer
          - WSI sampling level (zoom magnification)
        * - ``tile_size``
          - integer
          - Tiles are square images with ``tile_size`` x ``tile_size`` size in pixels.
        * - ``center_size``
          - integer
          - Size of the center of the tile. If the square in the centre is annotated, the tile is considered cancerous.
        * - ``include_annot_keywords``
          - array of strings
          - A list of keywords from an XML annotation of a WSI that designate the cancerous areas (required only by FROC)

``dirs`` Block
^^^^^^^^^^^^^^
    Defines paths to various data directories and to dataset.

    Accepts both relative and absolute paths. Pay attention to description where each relative path starts.

    ``results``

        A directory that will contain run results. Can be relative to project directory. Default value is ``models``.


        Note: The path is relative to the project root directory, from where the pipeline is run.

            *Type:* string

    ``root``

        A data set directory. Can be relative to project directory.

        Note: The path is relative to the project directory, from where the pipeline is run.

            *Type:* string

    ``rgb``

        Directory containing data from which the feature vectors are to be extracted.
        Typically contains the slides or symlins to them.

        Note: The path is relative to ``<project_dir>/data/``.

            *Type:* string

            *Default:* rgb

    ``label``

        Directory containing labels.

        Note: The path is relative to ``<root>`` directory.

            *Type:* string

            *Default:* label

    ``coord_maps``

        Directory containing "coordinate maps" (gzipped Pandas DataFrames with training examples).
        Can be omitted if the folder stem is the same as for the used dataset file.

        Note: The path is relative to ``<root>/coord_maps/`` directory.

            *Type:* string

    ``dataset`` Block

        ``name``

            Path to the dataset file. Either leads to an HDF5 file that contains paths to coordinate maps to a single pandas data frame that contains rows of all coordinate maps. HDF5 file should contain at least 'train' HDF5 Dataset within.

            Note: The path is relative to ``<root>/datasets/``

        ``valid_size``

            Defines the fraction (float) or an absolute number (integer) of examples in validation data split.

                *Default* 0.1

        ``test_size``

            Defines the fraction (float) or an absolute number (integer) of examples in test data split.

                *Default* 0.2


        ``pd_label_col``

            Optional key. Can be used to choose which column from coordinate maps should be used by an extractor class as a label in the current run. (Suitable if there are multiple lables available.)
            As an example, the prostate use case utilizes this option. Note that a user has to implement it in his or her own extractor.

                *Type* string

        .. note::
            The two types of dataset files approach stratification differently.

                HDF5: Splits can be stratified, if HDF5 file contains ``stratify`` atribute which is an array of integers (representing the classes) of the same length as the number of records.

                pandas DataFrame: Data splits can be stratified by adding ``stratify_col`` key to this configuration block. The value needs to be an existing column name from the data frame. The column's values will be used as stratification classes.

.. ==== GENERATOR==================

.. _generator:

``generator`` Block
```````````````````
    Specifications for data generator and runtime preprocessing.

``batch_size``
^^^^^^^^^^^^^^
    Size of batch.

        *Type:* integer

``valid_generator_type``
^^^^^^^^^^^^^^^^^^^^^^^^
    Generator type for validation data set.

        *Type:* string

        *Options:*

            * ``linear``
            * ``random``


``steps_per_epoch``
^^^^^^^^^^^^^^^^^^^
    Random generator samples ``batch_size * steps_per_epoch`` examples in each epoch.

        *Type:* integer


``validation_steps``
^^^^^^^^^^^^^^^^^^^^
    Number of validation steps.

        *Type:* integer

``train_augment``
^^^^^^^^^^^^^^^^^
    Whether to use augmentation on training examples.

        *Type:* boolean

``valid_augment``
^^^^^^^^^^^^^^^^^
    Whether to use augmentation on validation examples.

        *Type:* boolean


``test_augment``
^^^^^^^^^^^^^^^^
    Whether to use augmentation on testing examples.

        *Type:* boolean


``extractor`` Block
^^^^^^^^^^^^^^^^^^^
    Class that extracts a batch of data and transforms it into training and inference examples.
    Typically, each data set type needs its own Extractor class.


    ``class_name``
        A name of a class from module ``rationai.datagens.extractors``, where custom extractors can be implemented.

            *Type:* string

            *Options:*

                * ``BinaryClassExtractor``
                * ``SegmentationExtractor``


``augmenter`` Block
^^^^^^^^^^^^^^^^^^^
    Used by an extractor to augment extracted data.
    However, any transformation which does not change the data shape is allowed.

    ``class_name``
        A name of a class defined in ``rationai.datagens.augmenters``

        *Options:*
            * ``SlideAugmenter``

    ``config``
        A class specific block that defines augmentation/transformation parameters.
        See the concrete class' implementation for detailed information.

``sampler``
^^^^^^^^^^^
    Defines sampling behaviour and strategy.

    ``sequence_column``
        Defines which coordinate map's column should be considered as a data *sequence*.

        Model evaluation and inference is typically done in a *sequential* fashion e.g. slide by slide.
        Therefore, the ``SequentialSampler`` builds a sampling tree that aggregates examples by the chosen *sequence*, so their evaluations are separate.

            *Type:* string

    ``sampling_strat`` Block
        Specification on how to build a sampling tree from coordinate maps - pd.DataFrame(s).
        The keys represent the columns and their order in which they are randomly chosen during sampling.

        Additionaly, probability weights can be used either for class balancing or as a filtering mechanism.

        Example:
            Consider the following DataFrame as a coordinate map.

            .. image:: _static/images/df.png
                :width: 450

            Then for the configuration below, the sampler builds a tree where:

                #. A ``label`` is picked randomly using probability weights: 0.2 for False, 0.8 for True
                #. A ``slide_id`` is picked randomly using equal probability weights (``[]``)
                #. The final sampling is done from the sub DataFrame found in the chosen leaf node.

            .. code-block:: json

                {
                    "sampling_strat": {
                        "label": [0.2, 0.8],
                        "slide_id": []
                    }
                }

            Illustration of the sampling tree.

            .. image:: _static/images/sampling_tree_with_prob.png
                :width: 450

            .. note::
                Filtering can be done by binning a column values and then giving weights to the bin probabilities.
                e.g. column ``tissue_coverage`` is binned into two bins based on some minimum tissue threshold per tile.
                Then giving sampling probabilities [0.0, 1.0] to these bins achieves that the tiles with little tissue are never picked.

            .. warning::
                Probabilities are recommended only at the top level of the tree.
                At deeper levels, there is no guarantee that the probabilities can hold.
                This is due to the fact that some paths might not exists in the tree if they are not present in the data.


.. ==== MODEL=====================


.. _model:

``model`` Block
```````````````
    Specifications of the used model, checkpoints and metrics.

``class_name``
^^^^^^^^^^^^^^
    A class name of a model from ``rationai.training.models.tf_classifiers`` module.

        *Type:* string

        *Options:*

            * ``PretrainedModel``
            * ``UNet``
            * ``FCN8``


``keras_application``
^^^^^^^^^^^^^^^^^^^^^
    A serialized *function* (not module) from `tf.keras.applications <https://www.tensorflow.org/api_docs/python/tf/keras/applications>`_.
    Is only relevant if the ``class_name`` model requires a pretrained architecture. e.g., ``PretrainedModel``

    Example:
        Note that the config attribute ``include_top`` is False by default.

        .. code-block:: json

            {
                "class_name": "VGG16",
                "config": {
                    "pooling": "max"
                }
            }

``trainable``
^^^^^^^^^^^^^
    Sets if the underlying keras_application is trainable.

        *Type:* boolean

``optimizer``
^^^^^^^^^^^^^
    A serialized optimizer class from `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/>`_.

        *Type:* dict

        Example:

            .. code-block:: json

                {
                    "optimizer": {
                    "class_name": "Adam",
                        "config": {
                            "lr": 5e-6
                        }
                    }
                }

``loss``
^^^^^^^^
    Either a serialized loss class from `tf.keras.losses <https://www.tensorflow.org/api_docs/python/tf/keras/losses/>`_.
    or a custom loss from ``rationai.training.losses`` serialized in the same fashion as Keras' losses.

        *Type:* dict

        Example:

            .. code-block:: json

                {
                    "loss": {
                    "class_name": "BinaryCrossentropy"
                    }
                }

``regularizer``
^^^^^^^^^^^^^^^
    A serialized regularizer class from `tf.keras.regularizers <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/>`_.

        *Type:* dict

        Example:

            .. code-block:: json

                {
                    "regularizer": {
                    "class_name": "L2",
                    "config": {
                        "l2": 5e-05
                        }
                    }
                }


``input_shape``
^^^^^^^^^^^^^^^
    Shape of an input layer.

        *Type:* array

``output_size``
^^^^^^^^^^^^^^^
    Number of neurons in the final classification layer.

    Irrelevant for non-classification models.

        *Type:* integer

``train_checkpoint``
^^^^^^^^^^^^^^^^^^^^
    A name or path to a weight checkpoint.
    Checkpoint is loaded after model initialization.

        *Type:* string

.. _test_ckpt:

``test_checkpoint``
^^^^^^^^^^^^^^^^^^^
    A name or a path to a weight checkpoint.
    Checkpoint is loaded when the model enters a test mode (evaluation(.

    Key's value interpretation and look-up order:
        #. a file name in the *checkpoints directory* (*<project_dir>/models/<experiment_id>/callbacks/ModelCheckpoint/*)
        #. a relative path starting in *<project_dir>/*
        #. full path to arbitrary location

        *Type:* string

``output_bias``
^^^^^^^^^^^^^^^
    Initializer for the bias vector in the last dense layer of classfication models.

    See `tf.keras.layers.Dense <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_

``dropout``
^^^^^^^^^^^
    Configuration of the `Dropout <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout>`_ layer for classification models.

    *Type:* int or dict with kwargs

``sequence_metrics``
^^^^^^^^^^^^^^^^^^^^
    Per-sequence metrics that reset their state after each training epoch, or after each evaluated *sequence* during inference.
    Supports classes from `tf.keras.metrics <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/>`_ as well as custom classes defined in ``rationai.training.metrics``.

        *Type:* Array

        Example:

            .. code-block:: json

                {
                    "sequence_metrics": [
                        {"class_name": "BinaryAccuracy"},
                        {"class_name": "F1"}
                    ]
                }

``total_metrics``
^^^^^^^^^^^^^^^^^
    Per-dataset metrics that track overall performance across all testing sequences.
    Supports classes from `tf.keras.metrics <https://www.tensorflow.org/api_docs/python/tf/keras/metrics/>`_ as well as custom classes defined in ``rationai.training.metrics``.

        *Type:* Array

        Example:

            .. code-block:: json

                {
                    "total_metrics": [
                        {"class_name": "Recall"},
                        {"class_name": "Specificity"}
                    ]
                }


.. ==== STEP DEFINITIONS==========

.. _stepdefs:

``step_definitions`` Block
``````````````````````````
    A dictionary with *definitions* of pipeline's steps defined in ``ordered_steps`` inside :ref:`experiment<experiment>` block.

    A *step* is any class which inherits from ``StepInterface`` and implements ``from_params`` method with the required attributes (see the source code for exact specifications).
    Only ``StepInterface`` subclasses can be runnable in this fashion.

    Each step definition has two parts:

        * ``init`` is responsible for class instantiation.
            * The class is identified by ``class_id``.
            * If needed, additional init keyword arguments can be defined inside ``config``.

        * ``exec`` defines which ``method`` should be executed. If the method takes any parameters, they are specified inside ``kwargs`` block as keyword argumets.


        Example:

        .. code-block:: json

                {
                    "saliency": {
                        "init": {
                            "class_id": "rationai.visual.explain.SaliencyRunner",
                            "config": {
                                "grad_modifier": null
                            }
                        },
                        "exec": {
                            "method": "run",
                            "kwargs": {
                                "slide_names": ["TP-2019_6785-13-1"]
                            }
                        }
                    }
                }

        .. note::
            If multiple steps belong to one instance (see :ref:`context <context>`), only the first one is required to contain ``init`` specifications.


.. _train_test_notes:

Training & Inference Notes
``````````````````````````

    ``ExperimentRunner`` is a wrapper class that allows running training and inference as *steps*
    by handling cooperation of ``DataSource``, ``Datagen`` and the model.

    Its step definition requires ``experiment_class`` attribute inside the ``config`` block of the ``init`` part as can be seen below.
    This attribute defines which class from ``rationai.training.experiments`` will handle the training and evaluation.

        .. code-block:: json

            {
                "step_definitions": {
                    "exp.train": {
                        "init": {
                            "class_id": "rationai.training.ExperimentRunner",
                            "config": {
                                "experiment_class": "ClassificationExperiment"
                            }
                        }
                    }
                }
            }

Training
^^^^^^^^
    Training is done by running ``ExperimentRunner.train`` method,
    which accepts a single parameter - dictionary ``fit_kwargs``.
    Its contents are then passed as keyword arguments to Keras
    model's ``fit`` `method <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_
    (see example :ref:`example configuration <example_cfg>`).



Inference
^^^^^^^^^
    Inference is done by running ``ExperimentRunner.infer`` method.
    Similarly to training, the keyword arguments are specified & passed
    as a dictionary - ``infer_params``.
    However, there are three inference types, which are specified
    by the second parameter - ``infer_type``.

        * ``predict`` - calls ``model.predict`` - computes predictions but does not track the metrics (`see predict <https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict>`_).
        * ``evaluate`` - calls ``model.evaluate`` - computes metrics but does not keep predictions (`see evaluate <https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate>`_).
        * ``predict_evaluate`` - custom overriden method which combines both approaches



If serialized
`callbacks <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/>`_
are present in the training or inference configuration,
they get deserialized and are passed as objects.
If `ModelCheckpoint <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint>`_
callback is used, then file will be stored in the *checkpoint directory* (see :ref:`test_checkpoint <test_ckpt>`)
All other callbacks that write to disk will store their results systematically in *<project_dir>/models/<experiment_id>/callbacks/*. There, a separate folder will be created for each such callback using its class name.