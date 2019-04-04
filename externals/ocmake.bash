ocmake-source(){   echo $BASH_SOURCE ; }
ocmake-vi(){       vim $(ocmake-source) ; }
ocmake-env(){      olocal- ; opticks- ;  }
ocmake-usage(){ cat << EOU

CMake
=======

Typically it is preferable to use the cmake provided
by your system package manager.  However if that is not 
new enough, you can use the below bash functions to 
get and install a newer cmake.

Before doing so you can remove the system one with::

    sudo apt purge --auto-remove cmake     ## on Ubuntu 

Then get the new one with::

    ocmake-;ocmake--

EOU
}

ocmake-vers(){ echo 3.14.1 ; }
ocmake-nam(){ echo cmake-$(ocmake-vers) ; }
ocmake-url(){ echo https://github.com/Kitware/CMake/releases/download/v$(ocmake-vers)/cmake-$(ocmake-vers).tar.gz ; }
ocmake-dir(){ echo $LOCAL_BASE/opticks/externals/cmake/$(ocmake-nam) ; }
ocmake-cd(){  cd $(ocmake-dir) ; } 

ocmake-info(){ cat << EOI
$FUNCNAME
============

ocmake-vers : $(ocmake-vers)
ocmake-nam  : $(ocmake-nam)
ocmake-url  : $(ocmake-url)
ocmake-dir  : $(ocmake-dir)

EOI
}

ocmake-get(){
   local dir=$(dirname $(ocmake-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(ocmake-url)
   local tgz=$(basename $url)
   local nam=$(ocmake-nam)
   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz
}

ocmake-install()
{
   ocmake-cd
    ./bootstrap && make && sudo make install
}

ocmake--()
{
   ocmake-get
   ocmake-install
}


