Updating this documentation
============================

* https://simoncblyth.bitbucket.io/opticks/

Edit the RST::

    vi ../index.rst opticks.rst material_and_surface_properties.rst

Build the html from RST::

    [blyth@localhost ~]$ cd ~/opticks
    [blyth@localhost opticks]$ make
    ## fix any major RST problems 

Open the html documentation in your browser locally::

    epsilon:opticks blyth$ open ~/simoncblyth.bitbucket.io/opticks/index.html
    ## tip for gnome users, add open function to .bashrc :  open(){ gio open $* ; }

Publish html to bitbucket::

    [blyth@localhost ~]$ cd ~/simoncblyth.bitbucket.io/
    [blyth@localhost simoncblyth.bitbucket.io]$ git status 

    ...    ## git add new and modified files    

    [blyth@localhost simoncblyth.bitbucket.io]$ git commit -m "update docs, especially wrt CMake version requirement of 3.12+ "
    [blyth@localhost simoncblyth.bitbucket.io]$ git push 
    ...

   
Check the published result::

    open https://simoncblyth.bitbucket.io/opticks/





