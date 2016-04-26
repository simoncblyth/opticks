# === func-gen- : boost/boost fgp boost/boost.bash fgn boost fgh boost
boost-src(){      echo boost/boost.bash ; }
boost-source(){   echo ${BASH_SOURCE:-$(env-home)/$(boost-src)} ; }
boost-vi(){       vim $(boost-source) ; }
boost-env(){      elocal- ; }
boost-usage(){ cat << EOU

BOOST
=======

* http://www.boost.org/
* http://www.boost.org/users/history/


FindBoost.cmake
----------------

::

    simon:defaults blyth$ mdfind FindBoost.cmake
    /Users/blyth/env/cmake/Modules/FindBoost.cmake
    /opt/local/share/cmake-3.4/Modules/FindBoost.cmake
    /usr/local/env/graphics/oglplus/oglplus-0.59.0/config/FindBoost.cmake


installed versions
-------------------
   
G  1.49.0
C  1.32.0-7.rhel4 
N  1.33.1


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
boost-dir(){ echo $(local-base)/env/boost/$(boost-name) ; }
boost-bdir(){  echo $(boost-dir).build ; }
boost-prefix(){  echo $(boost-dir).local ; }

boost-ver(){ 
    #echo nuwa
    echo 1.54.0 
}

boost-nuwa-plat(){
  case $NODE_TAG in
     N) echo i686-slc5-gcc41-dbg ;;
     C) echo i686-slc4-gcc34-dbg ;;
  esac
}
boost-libdir(){  
   case $(boost-ver) in 
     nuwa) echo $DYB/external/Boost/1.38.0_python2.7/$(boost-nuwa-plat)/lib ;;
        *) echo  $(boost-prefix)/lib ;;
   esac
}
boost-incdir(){  
   case $(boost-ver) in 
    nuwa) echo $DYB/external/Boost/1.38.0_python2.7/$(boost-nuwa-plat)/include/boost-1_38 ;; 
       *) echo  $(boost-prefix)/include ;; 
   esac
}

boost-python-lib(){
   case $(boost-ver) in 
     nuwa) echo $(boost-python-lib-nuwa) ;; 
        *) echo boost_python ;; 
   esac
}

boost-python-lib-nuwa(){
  case $NODE_TAG in 
     N) echo boost_python-gcc41-mt ;;
     C) echo boost_python-gcc34-mt ;;
  esac   
}


boost-cd(){  cd $(boost-dir); }
boost-mate(){ mate $(boost-dir) ; }


boost-name(){ 
  case $(boost-ver) in 
    1.54.0) echo boost_1_54_0 ;; 
  esac       
}
boost-url(){ 
  case $(boost-ver) in 
    1.54.0) echo http://downloads.sourceforge.net/project/boost/boost/1.54.0/boost_1_54_0.tar.gz ;; 
  esac
}

boost-get(){
   local dir=$(dirname $(boost-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(boost-url)
   local nam=$(boost-name)
   local tgz=$(basename $url)
   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz
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

boost-bootstrap-help(){
  boost-cd
  ./bootstrap.sh --help
}

boost-bootstrap-build(){
  boost-cd
  ./bootstrap.sh --prefix=$(boost-prefix) --with-libraries=python
}

boost-build(){
  boost-cd
  ./b2 --build-dir=$(boost-bdir) install
}

boost-install-dir() {
    case $NODE_TAG in
        D) echo /opt/local ;;
        LT) echo /home/ihep/data/doc/home/ihep/juno-dev-new/ExternalLibs/Boost/1.55.0/ ;;
        GTL) echo /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/Release/J15v1r1/ExternalLibs/Boost/1.55.0/ ;;
        *) echo $(local-base);;
    esac
}

boost-idir() {
    boost-install-dir
}

boost-suffix() {
    case $NODE_TAG in
        D) echo '-mt' ;;
        *) echo ;;
    esac
}

boost-export() {
    export BOOST_INSTALL_DIR=$(boost-install-dir)
    export BOOST_SUFFIX=$(boost-suffix)
}

