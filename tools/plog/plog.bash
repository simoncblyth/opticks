# === func-gen- : tools/plog/plog fgp tools/plog/plog.bash fgn plog fgh tools/plog
plog-src(){      echo tools/plog/plog.bash ; }
plog-source(){   echo ${BASH_SOURCE:-$(env-home)/$(plog-src)} ; }
plog-vi(){       vi $(plog-source) ; }
plog-usage(){ cat << EOU


::

    In file included from /Users/blyth/env/numerics/npy/numpy.hpp:40:
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/fstream:864:20: error: no member named 'plog' in
          'std::__1::codecvt_base'; did you mean simply 'plog'?
            if (__r == codecvt_base::error)
                       ^

Resolve by moving the BLog.hh include after NPY.hpp::

     10 #include "NPY.hpp"
     12 #include "BLog.hh"


EOU
}
plog-env(){      opticks- ;  }
plog-dir(){ echo $(opticks-prefix)/externals/plog ; }
plog-cd(){  cd $(plog-dir); }
plog-url(){  echo https://github.com/SergiusTheBest/plog ; }
plog-get(){
   local dir=$(dirname $(plog-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d plog ] && git clone $(plog-url) 

}

plog-edit(){  vi $(opticks-home)/cmake/Modules/FindPLog.cmake ; }


plog-genlog-cc(){ 

   local tag=${1:-NPY}
   cat << EOL

#include <plog/Log.h>

#include "${tag}_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void ${tag}_LOG::Initialize(void* whatever, int level )
{
    PLOG_INIT(whatever, level);
}
void ${tag}_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

EOL
}

plog-genlog-hh(){ 
   local tag=${1:-NPY}
   cat << EOL

#pragma once
#include "${tag}_API_EXPORT.hh"

class ${tag}_API ${tag}_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

EOL
}

plog-genlog(){
  local msg="=== $FUNCNAME :"
  local ex=$(ls -1 *_API_EXPORT.hh 2>/dev/null) 
  [ -z "$ex" ] && echo $msg ERROR there is no export in PWD $PWD : run from project source with the correct tag : not $tag && return 

  local tag=${ex/_API_EXPORT.hh} 

  local cc=${tag}_LOG.cc
  local hh=${tag}_LOG.hh

  [ -f "$cc" -o -f "$hh" ] && echo $msg cc $cc or hh $hh exists already : delete to regenerate && return  

  echo $msg tag $tag generating cc $cc and hh $hh 

  plog-genlog-cc $tag > $cc
  plog-genlog-hh $tag > $hh

  echo $msg remember to commit and add to CMakeLists.txt 
}



