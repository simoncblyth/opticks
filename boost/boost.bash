boost-src(){      echo boost/boost.bash ; }
boost-source(){   echo ${BASH_SOURCE:-$(env-home)/$(boost-src)} ; }
boost-vi(){       vim $(boost-source) ; }
boost-usage(){ cat << EOU

BOOST
=======

* http://www.boost.org/
* http://www.boost.org/users/history/


Windows VS2015
--------------------

boost-build::

    ...failed updating 56 targets...
    ...skipped 4 targets...
    ...updated 14206 targets...

    ntuhep@ntuhep-PC MINGW64 /c/usr/local/opticks/externals/boost/boost_1_61_0


Rerunning indicates errors all due to lack of pyconfig.h.

::

    ntuhep@ntuhep-PC MINGW64 /c/usr/local/opticks/externals/boost/boost_1_61_0
    $ boost-b2 --show-libraries
    The following libraries require building:
        - atomic
        - chrono
        - container
        - context
        - coroutine
        - coroutine2
        - date_time
        - exception
        - filesystem
        - graph
        - graph_parallel
        - iostreams
        - locale
        - log
        - math
        - metaparse
        - mpi
        - program_options
        - python
        - random
        - regex
        - serialization
        - signals
        - system
        - test
        - thread
        - timer
        - type_erasure
        - wave


Building again with boost-b2-options for just the 
needed libs seems to succeed to build them but are not installed.
Reruning gives error::

    $ boost-build
    C:/usr/local/opticks/externals/boost/boost_1_61_0/tools/build/src/util\path.jam:461: in makedirs from module path
    error: Could not create directory '/c'


Rearranging the paths given to b2 to be windows style succeeds to install.


FindBoost.cmake
----------------

::

    simon:defaults blyth$ mdfind FindBoost.cmake
    /opt/local/share/cmake-3.4/Modules/FindBoost.cmake
    /usr/local/env/graphics/oglplus/oglplus-0.59.0/config/FindBoost.cmake


Linux
~~~~~~

Unfit for purpose::

    /usr/lib64/boost/BoostConfig.cmake
        
Use the cmake one::

    /home/blyth/local/env/tools/cmake/cmake-3.5.2-Linux-x86_64/share/cmake-3.5/Modules/FindBoost.cmake


Update Boost
-------------

::

    boost-bootstrap-build
    boost-build


Version History
-----------------

* http://www.boost.org/users/history/

::

    Version 1.60.0  December 17th, 2015 15:52 GMT
    Version 1.59.0  August 13th, 2015 15:23 GMT

    Version 1.44.0  August 13th, 2010 17:00 GMT
    Version 1.43.0  May 6th, 2010 12:00 GMT
    Version 1.42.0  February 2nd, 2010 14:00 GMT
    Version 1.41.0  November 17th, 2009 17:00 GMT


First releases
---------------

* http://www.boost.org/doc/libs/

::

    Asio              1.35.0
    System            1.35.0
    FileSystem        1.30.0 
    Log               1.54.0
    Program Options   1.32.0
    Property Tree     1.41.0
    


installed versions
-------------------

D  macports 1.59    
G  1.49.0
C  1.32.0-7.rhel4 
N  1.33.1
M  1.60.0-2

boost bootstrap build
-----------------------

::

    [blyth@ntugrid5 boost_1_60_0]$ boost-bootstrap-build
    Building Boost.Build engine with toolset gcc... tools/build/src/engine/bin.linuxx86_64/b2
    Detecting Python version... 2.6
    Detecting Python root... /usr
    Unicode/ICU support for Boost.Regex?... not found.
    Generating Boost.Build configuration in project-config.jam...

    Bootstrapping is done. To build, run:

        ./b2
        
    To adjust configuration, edit 'project-config.jam'.
    Further information:

       - Command line help:
         ./b2 --help
         
       - Getting started guide: 
         http://www.boost.org/more/getting_started/unix-variants.html
         
       - Boost.Build documentation:
         http://www.boost.org/build/doc/html/index.html




locally installed documentation
----------------------------------

open file:///opt/local/share/doc/boost/doc/html/index.html

* http://belle7.nuu.edu.tw/boost/libs/python/doc/building.html
* http://belle7.nuu.edu.tw/boost/more/getting_started/unix-variants.html

documentation system
~~~~~~~~~~~~~~~~~~~~~

https://svn.boost.org/trac/boost/wiki/BoostDocs/GettingStarted
https://svn.boost.org/trac/boost/wiki/DocsOrganization

sudo port install docbook-xml-4.2 docbook-xsl libxslt doxygen


show libraries
----------------

::

    [blyth@belle7 boost_1_54_0]$ ./bootstrap.sh --show-libraries
    Building Boost.Build engine with toolset gcc... tools/build/v2/engine/bin.linuxx86/b2

    The following Boost libraries have portions that require a separate build
    and installation step. Any library not listed here can be used by including
    the headers only.

    The Boost libraries requiring separate building and installation are:
        - atomic
        - chrono
        - context
        - coroutine
        - date_time
        - exception
        - filesystem
        - graph
        - graph_parallel
        - iostreams
        - locale
        - log
        - math
        - mpi
        - program_options
        - python
        - random
        - regex
        - serialization
        - signals
        - system
        - test
        - thread
        - timer
        - wave


bootstrap
------------

::

    [blyth@belle7 boost_1_54_0]$ type boost-bootstrap-build
    boost-bootstrap-build is a function
    boost-bootstrap-build () 
    { 
        boost-cd;
        ./bootstrap.sh --prefix=$(boost-prefix) --with-libraries=python
    }

    [blyth@belle7 boost_1_54_0]$ boost-bootstrap-build
    Building Boost.Build engine with toolset gcc... tools/build/v2/engine/bin.linuxx86/b2
    Detecting Python version... 2.7
    Detecting Python root... /data1/env/local/dyb/NuWa-trunk/../external/Python/2.7/i686-slc5-gcc41-dbg
    Unicode/ICU support for Boost.Regex?... not found.
    Generating Boost.Build configuration in project-config.jam...

    Bootstrapping is done. To build, run:

        ./b2
        
    To adjust configuration, edit 'project-config.jam'.
    Further information:

       - Command line help:
         ./b2 --help
         
       - Getting started guide: 
         http://www.boost.org/more/getting_started/unix-variants.html
         
       - Boost.Build documentation:
         http://www.boost.org/boost-build2/doc/html/index.html

boost python build
--------------------



EOU
}

boost-env(){      elocal- ; opticks- ;  }
boost-ver(){ 
    #echo 1.54.0 
    #echo 1.60.0 
    echo 1.61.0 
}
boost-name(){ local ver=$(boost-ver) ; echo boost_${ver//./_} ; }
boost-url(){ echo "http://downloads.sourceforge.net/project/boost/boost/$(boost-ver)/$(boost-name).tar.gz" ;  }


boost-fold(){   echo $(opticks-prefix)/externals/boost ; }
boost-dir(){    echo $(opticks-prefix)/externals/boost/$(boost-name) ; }
boost-prefix(){ echo $(opticks-prefix)/externals ; }

boost-bdir(){   echo $(boost-dir).build ; }
boost-idir(){   echo $(boost-prefix)/include/boost ; }

boost-cd(){   cd $(boost-dir); }
boost-bcd(){  cd $(boost-bdir); }
boost-icd(){  cd $(boost-idir); }
boost-fcd(){  cd $(boost-fold); }


boost-dir-old(){ echo $(local-base)/env/boost/$(boost-name) ; }
boost-prefix-old(){ echo $(boost-dir).local ; }

boost-get(){
   local iwd=$PWD
   local dir=$(dirname $(boost-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(boost-url)
   local nam=$(boost-name)
   local tgz=$(basename $url)

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxf $tgz

   cd $iwd
}



boost--() {
   boost-get
   boost-bootstrap-build
   boost-build
}


boost-bootstrap-help(){
  boost-cd
  ./bootstrap.sh --help
}

boost-bootstrap-build(){
  boost-cd

   case $(opticks-cmake-generator) in
     "Visual Studio 14 2015") cmd "/C bootstrap.bat" ;;
                           *) ./bootstrap.sh --prefix=$(boost-prefix) ;;  
   esac

  #./bootstrap.sh --prefix=$(boost-prefix) --with-libraries=python
}

boost-build(){
  boost-b2 $(boost-b2-options) install  
}


boost-gitbash2win(){
  local gbp=$1
  local wnp 
  case $gbp in
    /c/*) wnp=${gbp//\//\\}  ;;
       *) echo expecting gitbash style path starting with /c ;;
  esac
  echo "C:${wnp:2}"
}


boost-prefix-win(){ echo $(boost-gitbash2win $(boost-prefix)) ; }
boost-bdir-win(){   echo $(boost-gitbash2win $(boost-bdir)) ; }

boost-b2(){
   boost-cd

   case $(opticks-cmake-generator) in
     "Visual Studio 14 2015") cmd "/C b2 --prefix=$(boost-prefix-win) --build-dir=$(boost-bdir-win) $*  " ;;
                           *) ./b2 --prefix=$(boost-prefix) --build-dir=$(boost-bdir) $* ;;
   esac
}

boost-b2-options(){ cat << EOO
        --with-system
        --with-thread
        --with-program_options
        --with-log  
        --with-filesystem
        --with-regex
EOO
}



boost-example-(){ cat << EOE
#include <boost/lambda/lambda.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>

int main()
{
    using namespace boost::lambda;
    typedef std::istream_iterator<int> in;

    std::for_each(
        in(std::cin), in(), std::cout << (_1 * 3) << " " );
}
EOE
}
boost-example(){
  # when compiling from stdin need to specify the language eg with -x c++ 
  $FUNCNAME- | c++ -x c++ -I$(boost-incdir) - -o example
  echo 1 2 3 4 5 6 7 8 9 | ./example 
}







old-boost-install-dir() {
    case $NODE_TAG in
        D) echo /opt/local ;;
        LT) echo /home/ihep/data/doc/home/ihep/juno-dev-new/ExternalLibs/Boost/1.55.0/ ;;
        GTL) echo /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/Release/J15v1r1/ExternalLibs/Boost/1.55.0/ ;;
        *) echo $(local-base);;
    esac
}
old-boost-suffix() {
    case $NODE_TAG in
        D) echo '-mt' ;;
        *) echo ;;
    esac
}
old-boost-export() {
    export BOOST_INSTALL_DIR=$(boost-install-dir)
    export BOOST_SUFFIX=$(boost-suffix)
}

old-boost-nuwa-plat(){
  case $NODE_TAG in
     N) echo i686-slc5-gcc41-dbg ;;
     C) echo i686-slc4-gcc34-dbg ;;
  esac
}
old-boost-libdir(){  
   case $(boost-ver) in 
     nuwa) echo $DYB/external/Boost/1.38.0_python2.7/$(boost-nuwa-plat)/lib ;;
        *) echo  $(boost-prefix)/lib ;;
   esac
}
old-boost-incdir(){  
   case $(boost-ver) in 
    nuwa) echo $DYB/external/Boost/1.38.0_python2.7/$(boost-nuwa-plat)/include/boost-1_38 ;; 
       *) echo  $(boost-prefix)/include ;; 
   esac
}
old-boost-python-lib(){
   case $(boost-ver) in 
     nuwa) echo $(boost-python-lib-nuwa) ;; 
        *) echo boost_python ;; 
   esac
}
old-boost-python-lib-nuwa(){
  case $NODE_TAG in 
     N) echo boost_python-gcc41-mt ;;
     C) echo boost_python-gcc34-mt ;;
  esac   
}

