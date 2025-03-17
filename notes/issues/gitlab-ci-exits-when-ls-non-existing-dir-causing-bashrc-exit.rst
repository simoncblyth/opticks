gitlab-ci-exits-when-ls-non-existing-dir-causing-bashrc-exit
==============================================================



Seems gitlab has something against any command returning non-zero return code::

   set -eo pipefail




``set -e``
    Exit immediately if a pipeline (which may consist of a single simple command),  
    a subshell command enclosed in parentheses, or one of the commands
    executed as part of a command list enclosed by braces (see SHELL GRAMMAR
    above) exits with a non-zero status.  

    The shell does not exit if the command that fails:

    * is part of the command  list immediately following a while or until keyword, 
    * part of the test following the if or elif reserved words, 
    * part of any command executed in a && or || list except the command following the final && or ||, 
    * any command in a pipeline but the last, 
    * or if the command's return value is being inverted with !.  

    A trap on ERR, if set, is executed
    before  the shell  exits.   This  option  applies to the shell environment and
    each subshell environment separately (see COMMAND EXECUTION ENVIRONMENT above),
    and may cause subshells to exit before executing all the commands in the
    subshell.



``set -o pipefail``
    If set, the return value of a pipeline is the value of the last (rightmost)
    command to exit with a non-zero status, or zero if all commands in the pipeline
    exit successfully.  This option is disabled by default.


    

* https://gitlab.com/gitlab-org/gitlab-runner/-/issues/27668



is pipefail going to stymie the workaround ? 
-----------------------------------------------

* https://stackoverflow.com/questions/68668187/bash-set-e-not-exiting-immediately-with-pipefail

::

    || is a flow control operator, not a pipeline component. pipefail has no effect
    on it.

    If set -e caused flow control operators to exit, then you could never have an
    else branch of your script run with it active; it would be completely useless.

    For that reason, && and || suppress set -e behavior for their left-hand sides,
    just like if condition; then success; else fail; fi suppresses that behavior
    for condition.




gitlab-ci using "set -e" ? Problematic for the CMake mimicking function 
--------------------------------------------------------------------------




::

    opticks-setup-find-config-prefix () 
    { 
        : mimick CMake "find_package name CONFIG" identifing the first prefix in the path;
        local name=${1:-Geant4};
        local prefix="";
        local ifs=$IFS;
        IFS=:;
        for pfx in $CMAKE_PREFIX_PATH;
        do  
            ls -1 $pfx/lib*/$name-*/${name}Config.cmake 2> /dev/null 1>&2;
            [ $? -eq 0 ] && prefix=$pfx && break;
            ls -1 $pfx/lib*/cmake/$name-*/${name}Config.cmake 2> /dev/null 1>&2;
            [ $? -eq 0 ] && prefix=$pfx && break;
            ls -1 $pfx/lib*/cmake/$name/${name}Config.cmake 2> /dev/null 1>&2;
            [ $? -eq 0 ] && prefix=$pfx && break;
        done;
        IFS=$ifs;
        echo $prefix
    }



Try protection from "set -e"
-------------------------------

::

     opticks-setup-find-config-prefix(){
        : mimick CMake "find_package name CONFIG" identifing the first prefix in the path
        local name=${1:-Geant4}
        local prefix=""
        local rc=0
        local ifs=$IFS
        IFS=:
        for pfx in $CMAKE_PREFIX_PATH ; do
     
           : protect cmds that can give non-zero rc from "set -e" via pipeline but catch the rc 
           rc=1  
           ls -1 $pfx/lib*/$name-*/${name}Config.cmake 2>/dev/null 1>&2 && rc=$?
           [ $rc -eq 0 ] && prefix=$pfx && break
           
           ls -1 $pfx/lib*/cmake/$name-*/${name}Config.cmake 2>/dev/null 1>&2 && rc=$?
           [ $rc -eq 0 ] && prefix=$pfx && break
           
           ls -1 $pfx/lib*/cmake/$name/${name}Config.cmake 2>/dev/null 1>&2 && rc=$?
           [ $rc -eq 0 ] && prefix=$pfx && break
           
           # NB not general, doesnt find the lowercased form : but works for Geant4 and Boost 
        done 
        IFS=$ifs
        echo $prefix
     }







Symptom
--------

::

    cd /data1/blyth/local/opticks_Debug/Opticks-v0.3.2/x86_64--gcc11-geant4_10_04_p02-dbg


    A[blyth@localhost x86_64--gcc11-geant4_10_04_p02-dbg]$ source $PWD/bashrc
    A[blyth@localhost x86_64--gcc11-geant4_10_04_p02-dbg]$ opticks-setup-find-config-prefix Geant4
    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/Geant4/10.04.p02.juno
    A[blyth@localhost x86_64--gcc11-geant4_10_04_p02-dbg]$ 
    A[blyth@localhost x86_64--gcc11-geant4_10_04_p02-dbg]$ set -e
    A[blyth@localhost x86_64--gcc11-geant4_10_04_p02-dbg]$ opticks-setup-find-config-prefix Geant4
    Connection to 127.0.0.1 closed.
    epsilon:opticks blyth$ 



Others have met this and found workarounds
--------------------------------------------


* https://stackoverflow.com/questions/39466770/gitlab-ci-scripts-during-which-is-allowed-to-be-non-zero





