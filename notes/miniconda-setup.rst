miniconda-setup
===================


Install miniconda following

* https://www.anaconda.com/docs/getting-started/miniconda/install


Create "ok" environment for Opticks analysis::


    conda update -n base -c defaults conda    ## update base

    conda create -n ok                        

    conda info --envs                         ## list envs  

    conda activate ok

    conda info --envs                         ## list envs  



    conda install matplotlib

    conda install ipython

    conda install conda-forge::pyvista




Experience with recent ipython + matplotlib with Wayland
------------------------------------------------------------

Warning::

   qt.qpa.plugin: Count not find the Qt platform plugin "wayland" in ""

Below avoids the warning, and appears the same::

   export QT_QPA_PLATFORM=xcb 


Experience with recent ipython+pyvista 
----------------------------------------

Following plotting get SEGV on exit





