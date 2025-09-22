how_to_report_issues_and_investigate_opticks-full_fails
=========================================================



When reporting issues with installation or usage please include, via copy/paste where appropriate::

0. brief description of your environment : os etc.. and the instructions you are following
1. command line you are using
2. selected output showing the error or unexpected result that you got 

For extra points provide script/code/test that reproduces an issue.


Instructions you should be following
--------------------------------------

* https://simoncblyth.bitbucket.io/opticks/docs/opticks.html
* https://simoncblyth.bitbucket.io/opticks/docs/install.html



How to investigate "opticks-full" installation issues
-------------------------------------------------------

Installation will usually use the "opticks-full" bash function.
To pin down what part of "opticks-full" is failing it is useful to 
introspect the bash function with the "t" function, here introspecting itself::

    
    (ok) A[blyth@localhost ~]$ t t
    t () 
    { 
        typeset -f $*;
        : opticks.bash
    }


Here showing what "opticks-full" does::

    (ok) A[blyth@localhost ~]$ t opticks-full
    opticks-full () 
    { 
        local msg="=== $FUNCNAME :";
        local rc;
        opticks-info;
        [ $? -ne 0 ] && echo $msg ERR from opticks-info && return 1;
        opticks-full-externals;
        [ $? -ne 0 ] && echo $msg ERR from opticks-full-externals && return 2;
        opticks-full-make;
        [ $? -ne 0 ] && echo $msg ERR from opticks-full-make && return 3;
        opticks-install-extras;
        [ $? -ne 0 ] && echo $msg ERR from opticks-install-extras && return 4;
        opticks-cuda-capable;
        rc=$?;
        if [ $rc -eq 0 ]; then
            echo $msg detected GPU proceed with opticks-full-prepare;
            opticks-full-prepare;
            rc=$?;
            [ $rc -ne 0 ] && echo $msg ERR from opticks-full-prepare && return 5;
        else
            echo $msg detected no CUDA cabable GPU - skipping opticks-full-prepare;
            rc=0;
        fi;
        return 0
    }



Drilling down to "opticks-full-externals"::

    (ok) A[blyth@localhost ~]$ t opticks-full-externals
    opticks-full-externals () 
    { 
        local msg="=== $FUNCNAME :";
        echo $msg START $(date);
        local rc;
        echo $msg installing the below externals into $(opticks-prefix)/externals : eg bcm glm glfw glew gleq imgui plog nljson;
        opticks-externals;
        opticks-externals-install;
        rc=$?;
        [ $rc -ne 0 ] && return $rc;
        echo $msg config-ing the preqs : eg cuda optix;
        opticks-preqs;
        opticks-preqs-pc;
        rc=$?;
        [ $rc -ne 0 ] && return $rc;
        echo $msg config-ing the foreign : eg boost clhep xercesc g4;
        opticks-foreign;
        rc=$?;
        [ $rc -ne 0 ] && return $rc;
        echo $msg DONE $(date);
        return 0
    }
    (ok) A[blyth@localhost ~]$ 


The "opticks-externals" bash function just lists strings that correpond to 
bash functions that do the install::

    (ok) A[blyth@localhost ~]$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    plog
    nljson
    (ok) A[blyth@localhost ~]$ 


The installer::

    (ok) A[blyth@localhost ~]$ t opticks-externals-install
    opticks-externals-install () 
    { 
        opticks-installer- $(opticks-externals)
    }


Uses a pattern of eg "bcm-" to define functions then "bcm--" to do the main one, here downloading and installing::

    (ok) A[blyth@localhost ~]$ t opticks-installer-
    opticks-installer- () 
    { 
        echo $FUNCNAME;
        local msg="=== $FUNCNAME :";
        local pkgs=$*;
        local pkg;
        for pkg in $pkgs;
        do
            printf "\n\n\n############## %s ###############\n\n\n" $pkg;
            $pkg-;
            $pkg--;
            rc=$?;
            [ $rc -ne 0 ] && echo $msg RC $rc from pkg $pkg : ABORTING && return $rc;
        done;
        return 0
    }
    (ok) A[blyth@localhost ~]$ 


You can run those functions individually to investigate failed external installs eg::

      gleq-
      gleq--










