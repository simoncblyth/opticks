##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

boost-source(){   echo $BASH_SOURCE ; }
boost-vi(){       vim $(boost-source) ; }
boost-usage(){ cat << EOU

BOOST
=======

* http://www.boost.org/
* http://www.boost.org/users/history/


b2 docs
----------

* https://boostorg.github.io/build/manual/master/index.html#bbv2.faq.dll-path


Boost and Opticks
------------------

* boost is not listed in opticks-externals 
* boost is regarded as a pre-requisite 

  * huh : but it is not listed in opticks-preqs ?

* on macOS boost comes from macports ? 
 
  * huh : there is no boost.pc ?


Boost and CMake 
------------------

* :google:`Boost 1.70.0 CMake` shows lots of issues
* presumably Boost 1.70.0 has overhauled its CMake machinery.

Querying the CMake cache for boost variables ? From UseBoost bdir::

    [blyth@localhost build]$ cmake -LA -N . | grep Boost_
    Boost_DIR:PATH=Boost_DIR-NOTFOUND
    Boost_FILESYSTEM_LIBRARY_DEBUG:FILEPATH=/usr/lib64/libboost_filesystem-mt.so
    Boost_FILESYSTEM_LIBRARY_RELEASE:FILEPATH=/usr/lib64/libboost_filesystem-mt.so
    Boost_INCLUDE_DIR:PATH=/usr/include
    Boost_LIBRARY_DIR_DEBUG:PATH=/usr/lib64
    Boost_LIBRARY_DIR_RELEASE:PATH=/usr/lib64
    Boost_PROGRAM_OPTIONS_LIBRARY_DEBUG:FILEPATH=/usr/lib64/libboost_program_options-mt.so
    Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE:FILEPATH=/usr/lib64/libboost_program_options-mt.so
    Boost_REGEX_LIBRARY_DEBUG:FILEPATH=/usr/lib64/libboost_regex-mt.so
    Boost_REGEX_LIBRARY_RELEASE:FILEPATH=/usr/lib64/libboost_regex-mt.so
    Boost_SYSTEM_LIBRARY_DEBUG:FILEPATH=/usr/lib64/libboost_system-mt.so
    Boost_SYSTEM_LIBRARY_RELEASE:FILEPATH=/usr/lib64/libboost_system-mt.so
       
This means can know which Boost FindBoost found, but what is the source 
of truth ? How to influence the FindBoost at a higher level ? So can 
just use that as "truth", eg with::

   opticks-boost-libdir
   opticks-boost-includedir

Which present envvars.


* https://cmake.org/cmake/help/latest/command/find_package.html

Could use CMAKE_PREFIX_PATH envvar as the arbiter and as a standin for "truth", 
but that means need to duplicate the CMake Find logic ? 
Or just generate a simple CMakeLists.txt like cmak- and parse the output.
Too complicated. Something simple that works mostly is better than something 
complicated that always works : need to balance complexity and correctness. 




Boost and RPATH on CentOS7
------------------------------

Yum managed system boost has RPATH in which there are no boost libs ?::

    [blyth@localhost lib64]$ chrpath libboost_*.so
    libboost_atomic-mt.so: RPATH=/usr/lib:/usr/lib/python2.7/config
    libboost_atomic.so: RPATH=/usr/lib:/usr/lib/python2.7/config
    libboost_chrono-mt.so: RPATH=/usr/lib:/usr/lib/python2.7/config
    libboost_chrono.so: RPATH=/usr/lib:/usr/lib/python2.7/config
    libboost_context-mt.so: RPATH=/usr/lib:/usr/lib/python2.7/config

    /lib64 -> /usr/lib64






Boost and pkg-config : no offical solution
--------------------------------------------

* https://stackoverflow.com/questions/3971703/how-to-use-c-boost-library-with-pkg-config
* https://svn.boost.org/trac10/ticket/1094



On CentOS 7 got boost 1.53 from base repo 
---------------------------------------------

::

	[blyth@localhost imgui]$ yum list installed | grep boost 
	boost.x86_64                            1.53.0-27.el7                  @base    
	boost-atomic.x86_64                     1.53.0-27.el7                  @base    
	boost-chrono.x86_64                     1.53.0-27.el7                  @base    
	boost-context.x86_64                    1.53.0-27.el7                  @base    
	boost-date-time.x86_64                  1.53.0-27.el7                  @base    
	boost-devel.x86_64                      1.53.0-27.el7                  @base    
	boost-filesystem.x86_64                 1.53.0-27.el7                  @base    
	boost-graph.x86_64                      1.53.0-27.el7                  @base    
	boost-iostreams.x86_64                  1.53.0-27.el7                  @base    
	boost-locale.x86_64                     1.53.0-27.el7                  @base    
	boost-math.x86_64                       1.53.0-27.el7                  @base    
	boost-program-options.x86_64            1.53.0-27.el7                  @base    
	boost-python.x86_64                     1.53.0-27.el7                  @base    
	boost-random.x86_64                     1.53.0-27.el7                  @base    
	boost-regex.x86_64                      1.53.0-27.el7                  @base    
	boost-serialization.x86_64              1.53.0-27.el7                  @base    
	boost-signals.x86_64                    1.53.0-27.el7                  @base    
	boost-system.x86_64                     1.53.0-27.el7                  @base    
	boost-test.x86_64                       1.53.0-27.el7                  @base    
	boost-thread.x86_64                     1.53.0-27.el7                  @base    
	boost-timer.x86_64                      1.53.0-27.el7                  @base    
	boost-wave.x86_64                       1.53.0-27.el7                  @base    


Boost 1.48 Linux opticks/examples/UseUseBoost/PTreeIssue.cc
--------------------------------------------------------------

* https://stackoverflow.com/questions/47213341/does-boost-1-55-boostproperty-treeptree-compile-with-c11


Boost on iOS : build script
-----------------------------

* https://github.com/danoli3/ofxiOSBoost


Warnings regards symbol visibility
-------------------------------------

* :google:`boost symbol visibility hidden`
* http://stackoverflow.com/questions/15059360/compiling-boost-1-53-libraries-with-gcc-with-symbol-visibility-hidden

Why does this only show up in okc- ?

::

    simon:optickscore blyth$ okc--
    [  0%] Linking CXX shared library libOpticksCore.dylib
    ld: warning: direct access in boost::program_options::typed_value<std::__1::vector<int, std::__1::allocator<int> >, char>::value_type() const to global weak symbol typeinfo for std::__1::vector<int, std::__1::allocator<int> > means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in boost::typeindex::stl_type_index boost::typeindex::stl_type_index::type_id<std::__1::vector<int, std::__1::allocator<int> > >() to global weak symbol typeinfo for std::__1::vector<int, std::__1::allocator<int> > means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    [ 87%] Built target OpticksCore


* http://stackoverflow.com/questions/8685045/xcode-with-boost-linkerid-warning-about-visibility-settings

The linker complains about different visibility settings between your project and Boost.
You can also fix that issue by recompiling Boost with the same compatibility settings.
Just add to the bjam command line.::
  
   cxxflags=-fvisibility=hidden
   cxxflags=-fvisibility-inlines-hidden

   #    -fvisibility=hidden implies -fvisibility-inlines-hidden. Only the former is necessary. 


* https://svn.boost.org/trac/boost/ticket/6998

* https://gcc.gnu.org/wiki/Visibility



Boost Log Not Fit for Purpose
------------------------------

When boost is used from static libs, boost log does not work 
across DLLs. The symptoms are failure to set levels, missing msgs and crashes.

* https://sourceforge.net/p/boost-log/discussion/710022/thread/17cab0b7/

This issue prompted me to migrate opticks logging to plog-.

Also due to the crucial nature of logging that needs to 
work at every level of an application, I have formed the opinion:

* **regard logging as external to a project** 
* **do log setup via macros invoked from executable main**
* **NO integration of logging with application code, other than logging statements**

::


   LOG(info) << "hello" ;



Need to use Release for ITERATOR_DEBUGGING compatibility
-----------------------------------------------------------

::

    boost-bcd 

    $ find . -name '*.lib'
    ./boost/bin.v2/libs/atomic/build/msvc-14.0/debug/link-static/threading-multi/libboost_atomic-vc140-mt-gd-1_61.lib
    ./boost/bin.v2/libs/atomic/build/msvc-14.0/release/link-static/threading-multi/libboost_atomic-vc140-mt-1_61.lib
    ./boost/bin.v2/libs/chrono/build/msvc-14.0/debug/link-static/threading-multi/libboost_chrono-vc140-mt-gd-1_61.lib
    ./boost/bin.v2/libs/chrono/build/msvc-14.0/release/link-static/threading-multi/libboost_chrono-vc140-mt-1_61.lib
    ./boost/bin.v2/libs/container/build/msvc-14.0/debug/link-static/threading-multi/libboost_container-vc140-mt-gd-1_61.lib
    ./boost/bin.v2/libs/container/build/msvc-14.0/release/link-static/threading-multi/libboost_container-vc140-mt-1_61.lib
    ./boost/bin.v2/libs/context/build/msvc-14.0/debug/link-static/threading-multi/libboost_context-vc140-mt-gd-1_61.lib
    ./boost/bin.v2/libs/context/build/msvc-14.0/release/link-static/threading-multi/libboost_context-vc140-mt-1_61.lib


The release variants without gd already installed::

    ntuhep@ntuhep-PC MINGW64 /c/usr/local/opticks/externals/lib
    $ ll libboost_system*
    -rw-r--r-- 1 ntuhep 197121 619790 Jun 14 17:37 libboost_system-vc140-mt-gd-1_61.lib
    -rw-r--r-- 1 ntuhep 197121  80798 Jun 14 17:42 libboost_system-vc140-mt-1_61.lib





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

boost-env(){      olocal- ; opticks- ;  }
boost-ver(){ 
    #echo 1.54.0 
    #echo 1.60.0 
    #echo 1.61.0 
    echo  1.70.0
}
boost-name(){ local ver=$(boost-ver) ; echo boost_${ver//./_} ; }
boost-url(){ echo "http://downloads.sourceforge.net/project/boost/boost/$(boost-ver)/$(boost-name).tar.gz" ;  }


boost-fold(){   echo $(opticks-prefix)/externals/boost ; }
boost-dir(){    echo $(opticks-prefix)/externals/boost/$(boost-name) ; }
boost-prefix(){ echo $(opticks-prefix)/externals  ; }

boost-bdir(){   echo $(boost-dir).build ; }
boost-idir(){   echo $(boost-prefix)/include/boost ; }

boost-cd(){   cd $(boost-dir); }
boost-bcd(){  cd $(boost-bdir); }
boost-icd(){  cd $(boost-idir); }
boost-fcd(){  cd $(boost-fold); }


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
                           *) ./b2  hardcode-dll-paths=true dll-path=$(boost-prefix)/lib --prefix=$(boost-prefix) --build-dir=$(boost-bdir) $* ;;
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

boost-build(){
  boost-b2 $(boost-b2-options) install  
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



boost-components-(){ cat << EOC
system
program_options
filesystem
regex
EOC
}

boost-libs(){
   local comp
   boost-components- | while read comp ; do 
      printf -- "-lboost_%s " "$comp"
   done
}



boost-pc-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/boost.pc ; }
boost-pc-() 
{ 
   ## hmm need to branch depending on system boost or self-installed boost 

    cat <<EOP

# $FUNCNAME $(date)

prefix=$(opticks-prefix)
includedir=\${prefix}/externals/include
libdir=\${prefix}/externals/lib

Name: Boost
Description: 
Version: $(boost-ver)
Libs: -L\${libdir} $(boost-libs) -lstdc++ 
Cflags: -I\${includedir}

EOP
}

boost-pc() 
{ 
    local msg="=== $FUNCNAME :";
    local path=$(boost-pc-path);
    local dir=$(dirname $path);
    [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir;
    echo $msg $path;
    boost-pc- > $path
}



boost-rpath-fix-notes(){ cat << EON

https://stackoverflow.com/questions/33665781/dependencies-on-boost-library-dont-have-full-path/33893062#33893062

EON
}



boost-rpath-fix-Darwin(){
   local lib
   ls -1 libboost_*.dylib | while read lib ; do 
        install_name_tool $lib -id @rpath/$lib
   done
}
boost-rpath-fix-Linux(){
   local lib
   ls -1 libboost_*.so | while read lib ; do 
        chrpath --replace $PWD $lib
       # @rpath is a Darwinism ? 
   done
}
boost-rpath-fix(){
   cd $(opticks-prefix)/externals/lib

   if [ "$(uname)" == "Darwin" ]; then 
       boost-rpath-fix-$(uname) 
   else
       echo not running boost-rpath-fix-$(uname) 
   fi
}


boost-rpath-Darwin(){
   local lib
   ls -1 libboost_*.dylib | while read lib ; do 
        otool -L $lib
   done
}
boost-rpath-Linux(){
   local lib
   ls -1 libboost_*.so | while read lib ; do 
        chrpath $lib
   done
}
boost-rpath(){ 
   cd $(opticks-prefix)/externals/lib
   boost-rpath-$(uname) 
}

