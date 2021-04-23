Dependencies:

-   Python ofc
-   [pipenv](https://formulae.brew.sh/formula/pipenv)
-   [TinySol dataset](https://zenodo.org/record/3685367#.XnFp5i2h1IU%22)

To install, navigate to the package directory:

`$ pipenv install`

Generally, to train this model you will run:

`$ pipenv run train`

This first generates a dataset of targets from the samples contained in the TinySOL dataset. This will happen automatically when:

-   _src/settings.py_ does match the settings used to generate the target dataset.
-   a dataset has not already been generated.

In the event that you wish to manually generate a dataset before training, run:

`$ pipenv run retrain`

All of the model hyperparameters can be configured in _src/settings.py_.

When training is complete, the trained model is automatically saved in the _/models_ directory.
