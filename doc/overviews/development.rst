.. _dev:

Development
===========
The `Makefile` contains a few commands for development tasks such as running tests, checking formatting or installing the package.

For example, the following command would install the package and run tests::

        VENV_PATH='' make install test

If you would like to create a virtual environment and install to it, remove `VENV_PATH=''` from the above command.
