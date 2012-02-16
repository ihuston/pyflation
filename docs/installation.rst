************
Installation
************

Quick Installation
==================

Installing Pyflation is as simple as typing::

    $ python setup.py install

inside the directory where this file is located.

Requirements
============
Pyflation has the following requirements:

* Python 2.6 or higher (tested with 2.6.4) - http://www.python.org
* Numpy (tested with 1.3.0) - http://www.numpy.org
* Scipy (tested with 0.7.1) - http://www.scipy.org
* Cython (tested with 0.12.1) - http://www.cython.org
* PyTables 2.2 or higher (tested with 2.2) - http://www.pytables.org

The pyflation.cosmographs module contains helper functions to use 
with the Matplotlib package (http://matplotlib.sf.net) but this is not a
requirement of the core modules. 

In order to compile the documentation you will also need
* Sphinx (tested with 1.1.2) - http://http://sphinx.pocoo.org
* numpydoc (tested with 0.4) - http://pypi.python.org/pypi/numpydoc

Downloading Pyflation
=====================
Pyflation can be downloaded from the main website at http://pyflation.ianhuston.net or from
the Python Package Index at http://pypi.python.org/pypi/pyflation.

The latest development version is also available as a git and mercurial repository
at https://bitbucket.org/ihuston/pyflation/overview. 
For example to clone the git repository use::
    
    $ git clone git@bitbucket.org:ihuston/pyflation.git

Basic Installation
==================
 
First unpack the pyflation-x.x.x.tar.gz file in a suitable location. From the 
command line this can be done by using::

    $ tar xzvf pyflation-x.x.x.tar.gz

Then enter the created directory using::

    $ cd pyflation-x.x.x

The setup.py script will install pyflation in your local python installation. 
If you want to change the location of the installation list the possible options
using::

    $ python setup.py install --help

The most important option is --prefix which will change the base directory used
in deciding where to store the installation. The default depends on your 
installation and local python executable but is often /usr. For example if you 
want everything installed under the directory /home/me/localinstall/ then use::

    $ python setup.py install --prefix=/home/me/localinstall/

If you do not have the appropriate permissions you may not be able to install
to a system directory. In this case using the prefix option with a directory
you can write to is the best option.

Alternatively the virtualenv python module provides a useful way of keeping
python libraries in separate installations. This is useful if you work with
more than one version of a particular library and want to be able to run them
side by side. See http://virtualenv.openplans.org/ for more information.

Developer Installation
======================

If you do not want to install Pyflation into your system or local python 
site-packages directory, then it is possible to use it from within a separate
directory structure. Use the --copy-code flag and the --codedir option of 
pyflation-newrun.py to specify where the code directory is (possibly the current 
directory).
Execution of the other scripts should be done from within the new run directory,
i.e. where run_config.py resides, or the run directory should be added to the
PYTHONPATH environment variable. 

