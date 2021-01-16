Updating this documentation
============================

* https://simoncblyth.bitbucket.io/opticks/

Edit the RST::

    vi ../index.rst install.rst misc/material_and_surface_properties.rst

Build the html from RST::

    [blyth@localhost ~]$ cd ~/opticks
    [blyth@localhost opticks]$ make
    ## fix any major RST problems 

Open the html documentation in your browser locally (file based)::

    epsilon:opticks blyth$ open ~/simoncblyth.bitbucket.io/opticks/index.html
    ## tip for gnome users, add open function to .bashrc :  open(){ gio open $* ; }

Preview using local webserver::

    open http://localhost/opticks/index.html

In the rst reference images from ~/simoncblyth.bitbucket.io using urls 
starting with "//env" such as::

    .. image:: //env/Documents/Geant4OpticksWorkflow/Geant4OpticksWorkflow.001.png
        :width: 1024
        :alt: Geant4-Opticks-OptiX Workflow 


This works via the *env* and *opticks* symbolic links planted in `/Library/WebServer/Documents/` 

Publish html to bitbucket::

    [blyth@localhost ~]$ cd ~/simoncblyth.bitbucket.io/
    [blyth@localhost simoncblyth.bitbucket.io]$ git status 

    ...    ## git add new and modified files    

    [blyth@localhost simoncblyth.bitbucket.io]$ git commit -m "update docs, especially wrt CMake version requirement of 3.12+ "
    [blyth@localhost simoncblyth.bitbucket.io]$ git push 
    ...

   
Check the published result::

    open https://simoncblyth.bitbucket.io/opticks/



Tip for inclusion of RST that is maintained within sources
------------------------------------------------------------

::

    find . -name '*.rst' -exec grep -H start-after {} \;


cfg4/CTraverser.rst::

    .. include:: CTraverser.hh
       :start-after: /**
       :end-before: **/


