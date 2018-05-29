# === func-gen- : tools/plog/plog fgp externals/plog.bash fgn plog fgh tools/plog
plog-src(){      echo externals/plog.bash ; }
plog-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(plog-src)} ; }
plog-vi(){       vi $(plog-source) ; }
plog-usage(){ cat << EOU

PLOG : Simple header only logging that works across DLLs
============================================================

Inclusion of plog/Log.h brings in Windef.h that does::

   #define near 
   #define far

So windows dictates:

* you cannot have identifiers called "near" or "far"



::

    In file included from /Users/blyth/env/numerics/npy/numpy.hpp:40:
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/fstream:864:20: error: no member named 'plog' in
          'std::__1::codecvt_base'; did you mean simply 'plog'?
            if (__r == codecvt_base::error)
                       ^

Resolve by moving the BLog.hh include after NPY.hpp::

     10 #include "NPY.hpp"
     12 #include "BLog.hh"



fixed this by bringing my plog fork uptodate
-----------------------------------------------

:: 

    epsilon:externals blyth$ opticks--
    Scanning dependencies of target SysRap
    [  0%] Building CXX object sysrap/CMakeFiles/SysRap.dir/SYSRAP_LOG.cc.o
    In file included from /Users/blyth/opticks/sysrap/SYSRAP_LOG.cc:2:
    In file included from /usr/local/opticks/externals/plog/include/plog/Log.h:7:
    In file included from /usr/local/opticks/externals/plog/include/plog/Record.h:3:
    /usr/local/opticks/externals/plog/include/plog/Util.h:89:48: warning: 'syscall' is deprecated: first deprecated in macOS 10.12 - syscall(2) is
          unsupported; please switch to a supported interface. For SYS_kdebug_trace use kdebug_signpost(). [-Wdeprecated-declarations]
                return static_cast<unsigned int>(::syscall(SYS_thread_selfid));
                                                   ^
    /usr/include/unistd.h:745:6: note: 'syscall' has been explicitly marked deprecated here
    int      syscall(int, ...);
             ^
    1 warning generated.
    [  0%] Building CXX object sysrap/CMakeFiles/SysRap.dir/PLOG.cc.o
    In file included from /Users/blyth/opticks/sysrap/PLOG.cc:7:



but lastest plog has dangling else problem, have made pull request upstream
------------------------------------------------------------------------------

But Sergio didnt agree, tant pis : I continue to use my fork



Logging with embedded Opticks
-------------------------------

Current approach takes too much real estate in main with include + macro invoke
for every package you want to see logging from, see eg g4ok/test/G4OKTest.cc

How to shrink that ? Whats the biz with planting symbols in the separate proj libs ?

Can I make it invisible, controlled via envvar ?












EOU
}
plog-env(){      opticks- ;  }
plog-dir(){  echo $(opticks-prefix)/externals/plog ; }
plog-idir(){ echo $(opticks-prefix)/externals/plog/include/plog ; }
plog-ifold(){ echo $(opticks-prefix)/externals/plog/include ; }
plog-c(){    cd $(plog-dir); }
plog-cd(){   cd $(plog-dir); }
plog-icd(){  cd $(plog-idir); }


plog-url-upstream(){  echo https://github.com/SergiusTheBest/plog ; }
plog-url-pinned(){  echo https://github.com/simoncblyth/plog ; }
plog-url(){  plog-url-pinned ; }


plog-wipe(){
   local iwd=$PWD
   local dir=$(plog-dir)
   cd $(dirname $dir)
   rm -rf plog
   cd $iwd
}

plog-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(plog-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(plog-url)
   echo $msg url $url 
   [ ! -d plog ] && git clone $url
}

plog--()
{
   plog-get
}



plog-edit(){  vi $(opticks-home)/cmake/Modules/FindPLog.cmake ; }


plog-old-genlog-cc(){ 

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

plog-old-genlog-hh(){ 
   local tag=${1:-NPY}
   cat << EOL

#pragma once
#include "${tag}_API_EXPORT.hh"

#define ${tag}_LOG__ \
 { \
    ${tag}_LOG::Initialize(plog::get(), PLOG::instance->prefix_parse( info, "${tag}") ); \
 } \


#define ${tag}_LOG_ \
{ \
    ${tag}_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() ); \
} \


class ${tag}_API ${tag}_LOG {
   public:
       static void Initialize(void* whatever, int level );
       static void Check(const char* msg);
};

EOL
}









plog-genlog-hh(){ 
   local tag=${1:-NPY}
   cat << EOL

#pragma once
#include "${tag}_API_EXPORT.hh"

#define ${tag}_LOG__ \
 { \
    ${tag}_LOG::Initialize(PLOG::instance->prefixlevel_parse( info, "${tag}"), plog::get(), NULL ); \
 } \


#define ${tag}_LOG_ \
{ \
    ${tag}_LOG::Initialize(plog::get()->getMaxSeverity(), plog::get(), NULL ); \
} \

class ${tag}_API ${tag}_LOG {
   public:
       static void Initialize(int level, void* app1, void* app2 );
       static void Check(const char* msg);
};

EOL
}



plog-genlog-cc(){ 

   local tag=${1:-NPY}
   cat << EOL

#include <plog/Log.h>

#include "${tag}_LOG.hh"
#include "PLOG_INIT.hh"
#include "PLOG.hh"
       
void ${tag}_LOG::Initialize(int level, void* app1, void* app2 )
{
    PLOG_INIT(level, app1, app2);
}
void ${tag}_LOG::Check(const char* msg)
{
    PLOG_CHECK(msg);
}

EOL
}


plog-genlog(){
  local cmd=$1 
  local msg="=== $FUNCNAME :"
  local ex=$(ls -1 *_API_EXPORT.hh 2>/dev/null) 
  [ -z "$ex" ] && echo $msg ERROR there is no export in PWD $PWD : run from project source with the correct tag : not $tag && return 

  local tag=${ex/_API_EXPORT.hh} 
  local cc=${tag}_LOG.cc
  local hh=${tag}_LOG.hh

  if [ "$cmd" == "FORCE" ] ; then 
     rm -f $cc
     rm -f $hh
  fi

  [ -f "$cc" -o -f "$hh" ] && echo $msg cc $cc or hh $hh exists already : delete to regenerate && return  

  echo $msg tag $tag generating cc $cc and hh $hh 

  plog-genlog-cc $tag > $cc
  plog-genlog-hh $tag > $hh

  echo $msg remember to commit and add to CMakeLists.txt 
}


plog-inplace-edit(){
   perl -pi -e 's,BLog\.hh,PLOG.hh,g' *.cc && rm *.cc.bak
}


plog-t-(){ cat << EOC

#include <plog/Log.h> // Step1: include the header.

int main()
{
    plog::init(plog::debug, "Hello.txt"); // Step2: initialize the logger.

    // Step3: write log messages using a special macro. 
    // There are several log macros, use the macro you liked the most.

    LOGD << "Hello log!"; // short macro
    LOG_DEBUG << "Hello log!"; // long macro
    LOG(plog::debug) << "Hello log!"; // function-style macro

    int verbose = 5 ; 
    if(verbose > 3) LOGD << "logging in short causes dangling else " ; 
       

    return 0;
}

EOC
}

plog-t()
{
   local tmp=/tmp/$USER/opticks/$FUNCNAME
   mkdir -p $tmp
   cd $tmp
   local name=plogtest.cc
   $FUNCNAME- > $name
   cc $name -I$(plog-ifold) -lc++ -o plogtest
   ./plogtest 
   cat Hello.txt

}

plog-t-notes(){ cat << EON

Updating plog ?
=================

macOS 10.13.4 deprecated syscall
------------------------------------

::

    epsilon:plog-t blyth$ plog-;plog-t
    In file included from plogtest.cc:2:
    In file included from /usr/local/opticks/externals/plog/include/plog/Log.h:7:
    In file included from /usr/local/opticks/externals/plog/include/plog/Record.h:3:
    /usr/local/opticks/externals/plog/include/plog/Util.h:89:48: warning: 'syscall' is deprecated: first deprecated in macOS 10.12 - syscall(2) is
          unsupported; please switch to a supported interface. For SYS_kdebug_trace use kdebug_signpost(). [-Wdeprecated-declarations]
                return static_cast<unsigned int>(::syscall(SYS_thread_selfid));
                                                   ^
    /usr/include/unistd.h:745:6: note: 'syscall' has been explicitly marked deprecated here
    int      syscall(int, ...);
             ^
    1 warning generated.
    epsilon:plog-t blyth$ 


Offending line 89::

     82         inline unsigned int gettid()
     83         {
     84 #ifdef _WIN32
     85             return ::GetCurrentThreadId();
     86 #elif defined(__unix__)
     87             return static_cast<unsigned int>(::syscall(__NR_gettid));
     88 #elif defined(__APPLE__)
     89             return static_cast<unsigned int>(::syscall(SYS_thread_selfid));
     90 #endif
     91         }


Note changes in the latest plog:

* https://github.com/SergiusTheBest/plog/blob/master/include/plog/Util.h

::

    #elif defined(__APPLE__)
            uint64_t tid64;
            pthread_threadid_np(NULL, &tid64);
            return static_cast<unsigned int>(tid64);
    #endif



dangling else:  "if(smth) LOG(info) << blah "  
-------------------------------------------------

Old plog macros didnt have this issue

/Volumes/Delta/usr/local/opticks/externals/plog/include/plog/Log.h::

     28 //////////////////////////////////////////////////////////////////////////
     29 // Log severity level checker
     30 
     31 #define IF_LOG_(instance, severity)     if (plog::get<instance>() && plog::get<instance>()->checkSeverity(severity))
     32 #define IF_LOG(severity)                IF_LOG_(PLOG_DEFAULT_INSTANCE, severity)
     33 
     34 //////////////////////////////////////////////////////////////////////////
     35 // Main logging macros
     36 
     37 #define LOG_(instance, severity)        IF_LOG_(instance, severity) (*plog::get<instance>()) += plog::Record(severity, PLOG_GET_FUNC(), __LINE__, PLOG_GET_THIS())
     38 #define LOG(severity)                   LOG_(PLOG_DEFAULT_INSTANCE, severity)
     39 


/usr/local/opticks/externals/plog/include/plog/Log.h::

     34 // Log severity level checker
     35 
     36 #define IF_LOG_(instance, severity)     if (!plog::get<instance>() || !plog::get<instance>()->checkSeverity(severity)) {;} else
     37 #define IF_LOG(severity)                IF_LOG_(PLOG_DEFAULT_INSTANCE, severity)
     38 
     39 //////////////////////////////////////////////////////////////////////////
     40 // Main logging macros
     41 
     42 #define LOG_(instance, severity)        IF_LOG_(instance, severity) (*plog::get<instance>()) += plog::Record(severity, PLOG_GET_FUNC(), __LINE__, PLOG_GET_FILE(), PLOG_GET_THIS())
     43 #define LOG(severity)                   LOG_(PLOG_DEFAULT_INSTANCE, severity)
     44 


::

    36 //#define IF_LOG_(instance, severity)     if (!plog::get<instance>() || !plog::get<instance>()->checkSeverity(severity)) {;} else
     37 #define IF_LOG_(instance, severity)     if (plog::get<instance>() && plog::get<instance>()->checkSeverity(severity)) 

    epsilon:plog blyth$ git diff
    diff --git a/include/plog/Log.h b/include/plog/Log.h
    index cf0a68c..e45669c 100644
    --- a/include/plog/Log.h
    +++ b/include/plog/Log.h
    @@ -33,7 +33,8 @@
     //////////////////////////////////////////////////////////////////////////
     // Log severity level checker
     
    -#define IF_LOG_(instance, severity)     if (!plog::get<instance>() || !plog::get<instance>()->checkSeverity(severity)) {;} else
    +//#define IF_LOG_(instance, severity)     if (!plog::get<instance>() || !plog::get<instance>()->checkSeverity(severity)) {;} else^M
    +#define IF_LOG_(instance, severity)     if (plog::get<instance>() && plog::get<instance>()->checkSeverity(severity)) ^M
     #define IF_LOG(severity)                IF_LOG_(PLOG_DEFAULT_INSTANCE, severity)
     
     //////////////////////////////////////////////////////////////////////////
    epsilon:plog blyth$ 




EON
}


plog-issue-(){ cat << EOI


[  1%] Building CXX object sysrap/CMakeFiles/SysRap.dir/SMap.cc.o
/Users/blyth/opticks/sysrap/SMap.cc:26:9: warning: add explicit braces to avoid dangling else [-Wdangling-else]
        LOG(info) << " value " << std::setw(32) << std::hex << value << std::dec ; 
        ^
/usr/local/opticks/externals/plog/include/plog/Log.h:43:41: note: expanded from macro 'LOG'
#define LOG(severity)                   LOG_(PLOG_DEFAULT_INSTANCE, severity)
                                        ^
/usr/local/opticks/externals/plog/include/plog/Log.h:42:41: note: expanded from macro 'LOG_'
#define LOG_(instance, severity)        IF_LOG_(instance, severity) (*plog::get<instance>()) += plog::Record(severity, PLOG_GET_FUNC(), __LIN...
                                        ^
/usr/local/opticks/externals/plog/include/plog/Log.h:36:124: note: expanded from macro 'IF_LOG_'
#define IF_LOG_(instance, severity)     if (!plog::get<instance>() || !plog::get<instance>()->checkSeverity(severity)) {;} else
                                                                                                                           ^
/Users/blyth/opticks/sysrap/SMap.cc:35:13: warning: add explicit braces to avoid dangling else [-Wdangling-else]
            LOG(info) 


EOI
}


