Updating this documentation
============================

* https://simoncblyth.bitbucket.io/opticks/

Edit the RST::

    index.rst
    docs/opticks.rst
    ...


Build the html from RST::

    [blyth@localhost ~]$ cd ~/opticks
    [blyth@localhost opticks]$ make
    ## fix any major RST problems 

Publish html to bitbucket::

    [blyth@localhost ~]$ cd ~/simoncblyth.bitbucket.io/
    [blyth@localhost simoncblyth.bitbucket.io]$ hg st          

    ...    ## add new files    

    [blyth@localhost simoncblyth.bitbucket.io]$ hg commit -m "update docs, especially wrt CMake version requirement of 3.12+ "
    [blyth@localhost simoncblyth.bitbucket.io]$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/simoncblyth.bitbucket.io
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 1 changesets with 98 changes to 98 files

   
Check the published result::

    open https://simoncblyth.bitbucket.io/opticks/





