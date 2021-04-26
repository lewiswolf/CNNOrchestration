Dependencies:

-   Python ofc
-   [pipenv](https://formulae.brew.sh/formula/pipenv)
-   [TinySol dataset](https://zenodo.org/record/3685367#.XnFp5i2h1IU%22)

To install, navigate to the package directory:

`$ pipenv install`

Pytorch is installed automatically, and will work fine for all CPU based usages. However, to configure this package for GPU usage, you must install your required pytorch version via:

`$ pipenv run pip install torch==1.8.1+cu102 ...`

Generally, to train this model you will run:

`$ pipenv run train`

This first generates a dataset of targets from the samples contained in the TinySOL dataset. This will happen automatically when:

-   _src/settings.py_ does match the settings used to generate the target dataset.
-   a dataset has not already been generated.

In the event that you wish to manually generate a dataset before training, run:

`$ pipenv run retrain`

All of the model hyperparameters can be configured in _src/settings.py_. When training is complete, the trained model is automatically saved in the _/models_ directory.

To test the model, run:

`$ pipenv run orchestrate`

This will prompt the user for a filepath to a target sound, and generate an orchestration using a pretrained model defined in _src/settings.py_. Generated samples are output to _/out_.

If you wish to run the entire process, including generating the dataset (when necessary), training and evaluating, use:

`$ pipenv run pipeline`

Alternatively, use `$ pipenv run python index.py --help` for more details on using optional arguments.
