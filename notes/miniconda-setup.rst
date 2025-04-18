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

