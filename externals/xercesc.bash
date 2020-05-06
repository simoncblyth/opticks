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

# === func-gen- : xml/xercesc/xercesc fgp externals/xercesc.bash fgn xercesc fgh xml/xercesc
xercesc-src(){      echo externals/xercesc.bash ; }
xercesc-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(xercesc-src)} ; }
xercesc-vi(){       vi $(xercesc-source) ; }
xercesc-env(){      olocal- ; }
xercesc-usage(){ cat << EOU

XERCESC
========

* XML handling package required for Geant4 GDML support


On Centos 7 got from base repo
---------------------------------

::

	[blyth@localhost imgui]$ yum list installed | grep xerces
	xerces-c.x86_64                         3.1.1-8.el7_2                  @base    
	xerces-c-devel.x86_64                   3.1.1-8.el7_2                  @base    


FindEnvXercesC.cmake
----------------------

See *xercesc-edit*::

     21 # try to find the header
     22 FIND_PATH(XERCESC_INCLUDE_DIR xercesc/parsers/SAXParser.hpp 
     23   ${XERCESC_ROOT_DIR}/include
     24   HINTS ENV  XERCESC_ROOT
     25   /usr/include 
     26   /usr/local/include
     27   /opt/local/include
     28 )
     29 
     30 # Find the library
     31 FIND_LIBRARY(XERCESC_LIBRARY
     32    NAMES xerces-c 
     33    HINTS ENV  XERCESC_ROOT
     34    PATHS
     35      ${XERCESC_ROOT_DIR}/lib
     36      /usr/lib 
     37      /usr/local/lib
     38      /opt/local/lib
     39    DOC "The name of the xerces-c library"
     40 )



Multi-Version Confusion Issue
-------------------------------

* probably easiest to 


cfg4 link failing with xercesc_2_8::

    [ 71%] Linking CXX shared library libcfg4.dylib
    Undefined symbols for architecture x86_64:
      "xercesc_2_8::DTDEntityDecl::serialize(xercesc_2_8::XSerializeEngine&)", referenced from:
          vtable for xercesc_2_8::DTDEntityDecl in CGDMLDetector.cc.o
      "xercesc_2_8::XMLAttDefList::serialize(xercesc_2_8::XSerializeEngine&)", referenced from:
          vtable for xercesc_2_8::XMLAttDefList in CGDMLDetector.cc.o
      "xercesc_2_8::XMLEntityDecl::~XMLEntityDecl()", referenced from:
          xercesc_2_8::DTDEntityDecl::~DTDEntityDecl() in CGDMLDetector.cc.o
      "xercesc_2_8::SAXParseException::SAXParseException(xercesc_2_8::SAXParseException const&)", referenced from:
          xercesc_2_8::HandlerBase::fatalError(xercesc_2_8::SAXParseException const&) in CGDMLDetector.cc.o

G4 expecting libxerces-c-3.1.dylib::

    simon:lib blyth$ otool -L libG4persistency.dylib
    libG4persistency.dylib:
        @rpath/libG4persistency.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4run.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/local/opticks/lib/libxerces-c-3.1.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4event.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4tracking.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4processes.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4digits_hits.dylib (compatibility version 0.0.0, current version 0.0.0)
        /opt/local/lib/libexpat.1.dylib (compatibility version 8.0.0, current version 8.2.0)
        @rpath/libG4zlib.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4track.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4particles.dylib (compatibility version 0.0.0, current version 0.0.0)

::

    simon:cfg4 blyth$ port info xercesc
    xercesc @2.8.0_3 (textproc)
    Variants:             universal

    simon:cfg4 blyth$ port info xercesc3
    xercesc3 @3.1.4 (textproc, xml, shibboleth)
    Variants:             universal


    simon:tests blyth$ c++ XercescCTest.cc -I/opt/local/include -L/opt/local/lib -lc++
    Undefined symbols for architecture x86_64:
      "xercesc_2_8::XMLPlatformUtils::Initialize(char const*, char const*, xercesc_2_8::PanicHandler*, xercesc_2_8::MemoryManager*, bool)", referenced from:
          _main in XercescCTest-6063d6.o
      "xercesc_2_8::XMLUni::fgXercescDefaultLocale", referenced from:
          _main in XercescCTest-6063d6.o
    ld: symbol(s) not found for architecture x86_64
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    simon:tests blyth$ 
    simon:tests blyth$ 
    simon:tests blyth$ pwd
    /Users/blyth/opticks/cfg4/tests



EOU
}

xercesc-prefix-old(){  

  case $(uname -s) in 
      Darwin) echo /opt/local ;;
    MINGW64*) echo /mingw64 ;;
           *) echo ${LOCAL_BASE:-/usr/local}/opticks/externals ;;
  esac  
}

xercesc-prefix(){  
    echo $(opticks-prefix)/externals
}


xercesc-edit(){ vi $(opticks-home)/cmake/Modules/FindEnvXercesC.cmake ; }


xercesc-library-macports(){     echo /opt/local/lib/libxerces-c.dylib ; }
xercesc-include-dir-macports(){ echo /opt/local/include ; }

xercesc-library(){     echo ${OPTICKS_XERCESC_LIBRARY:-$(xercesc-library-)} ; }
xercesc-include-dir(){ echo ${OPTICKS_XERCESC_INCLUDE_DIR:-$(xercesc-include-dir-)} ; }

xercesc-library-(){  

  if [ "$NODE_TAG" == "MGB" ]; then
      ome-
      ome-xercesc-lib
  elif [ "$NODE_TAG" == "X" ]; then
      echo /lib64/libxerces-c-3.1.so    
  else 
      case $(uname -s) in 
           Darwin) echo $(xercesc-library-macports)  ;;
         MINGW64*) echo $(xercesc-prefix)/bin/libxerces-c-3-1.dll ;;
                *) echo $(xercesc-prefix)/lib/libxerces-c-3.1.so  ;;
      esac
  fi 
}

xercesc-include-dir-(){ 
  if [ "$NODE_TAG" == "MGB" ]; then
      ome-
      ome-xercesc-include
  else
      case $(uname -s) in 
           Darwin) echo $(xercesc-include-dir-macports)  ;;
                *) echo $(xercesc-prefix)/include    ;;
      esac
  fi
}


xercesc-geant4-export(){
  export XERCESC_INCLUDE_DIR=$(xercesc-include-dir)
  export XERCESC_LIBRARY=$(xercesc-library)
  export XERCESC_ROOT_DIR=$(xercesc-prefix)
}

xercesc-url(){ echo http://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.1.1.tar.gz ; }
xercesc-dist(){ echo $(basename $(xercesc-url)); }
xercesc-name(){ local dist=$(xercesc-dist) ; echo ${dist/.tar.gz} ; }
xercesc-base(){ echo $(opticks-prefix)/externals/xercesc ; }

xercesc-dir(){  echo $(xercesc-prefix)/xercesc/$(xercesc-name) ; }

xercesc-info(){ cat << EOI

$FUNCNAME
==============

USED BY CMAKE, FOR EITHER SYSTEM OR MANULLY INSTALLED XERCES-C

   xercesc-library : $(xercesc-library)
   xercesc-include-dir : $(xercesc-include-dir)

ONLY RELEVANT WHEN BUILDING MANUALLY 

   xercesc-url    : $(xercesc-url)
   xercesc-dist   : $(xercesc-dist)
   xercesc-name   : $(xercesc-name)
   xercesc-base   : $(xercesc-base)
   xercesc-dir    : $(xercesc-dir)

   xercesc-prefix  : $(xercesc-prefix)


   xercesc-library-macports : $(xercesc-library-macports)
   xercesc-include-dir-macports : $(xercesc-include-dir-macports)

   On Mac a preexisting xercesc from macports can confuse the build, 
   in this case probably better to just use macports xercesc
   see g4-cmake-modify-adopt-macports-xercesc


NB configure BASED BUILD 

* source and build dirs are not separated
* avoided configure being run at every invokation by checking for a Makefile
  in order to avoid rebuilding this 
* despite this G4persistency is still being rebuilt 
  everytime opticks-externals-install is run 

EOI
}



xercesc-cd(){   cd $(xercesc-dir); }

xercesc-get(){
   local dir=$(dirname $(xercesc-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(xercesc-url)
   local tgz=$(xercesc-dist)
   local nam=$(xercesc-name)
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxf $tgz 
}

xercesc-configure()
{
   local msg="=== $FUNCNAME :"
   xercesc-cd

   if [ -f $(xercesc-dir)/Makefile -a -f $(xercesc-dir)/xerces-c.pc ]; then
      echo $msg looks to have been configured already 
   else
      ./configure --prefix=$(xercesc-prefix)
   fi 
}

xercesc-make()
{
   xercesc-cd
   make ${1:-install}
}


xercesc-pc(){ 
   local msg="=== $FUNCNAME :"
   local paths="$OPTICKS_PREFIX/externals/lib/pkgconfig/xerces-c.pc /opt/local/lib/pkgconfig/xerces-c.pc"
   local path2="$OPTICKS_PREFIX/externals/lib/pkgconfig/OpticksXercesC.pc"
   local path
   for path in $paths ; do 
       if [ -f "$path" -a ! -f "$path2" ]; then
           echo $msg $path to $path2
           cp $path $path2  
           # no-fixing when comes from macports 
       elif [ -f "$path2" ]; then 
           echo $msg $path $path2 exists already
       fi 
   done
}

xercesc--()
{
   local msg="=== $FUNCNAME :"

   xercesc-info

   if [ "$(uname)" == "Darwin" ]; then 

       xercesc-darwin 

   elif [ "$(uname)" == "Linux" ]; then 
   
       xercesc-get
       [ $? -ne 0 ] && echo $msg get FAIL && return 1
       xercesc-configure
       [ $? -ne 0 ] && echo $msg configure FAIL && return 2
       xercesc-make
       [ $? -ne 0 ] && echo $msg make FAIL && return 3
   fi 

   xercesc-pc
   [ $? -ne 0 ] && echo $msg pc FAIL && return 4

   return 0 
}

xercesc-darwin(){ cat << EOD

$FUNCNAME : on OSX use macports xercesc 

EOD
}

xercesc-setup(){ cat << EOS
# $FUNCNAME
EOS
}

