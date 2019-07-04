ubuntu-bash-login-shell-differences
=====================================


ISSUE : opticks- command undefined with tboolean.sh that currently has "/bin/bash -l"
------------------------------------------------------------------------------------------

::

    Hi Simon,

    Thank you for the additional commits.

    Using the newest one I've got the interpolationTest to pass.

    Following the workflow geocache creation guide, what did geocache-create call (as that function no longer exists)?
    I assume that it's now geocache-recreate as this attempts to load JUNO (which happens with no problem).
    Doing this and then exporting the key (geocache-key-export) and then running tboolean-boxsphere- produces no errors.
    However the test will not pass as tboolean.sh still cannot doesn't recognise the functions.

    Additionally for tboolean.sh, my .bashrc already had 'opticks-' in it and the 'command not found' is still seen for all the subsequent functions (geocache- etc...).

    Best,
    Sam



Reproduce in VirtualBox Ubuntu 18.04.2
--------------------------------------------

.bashrc::

    001 # ~/.bashrc: executed by bash(1) for non-login shells.
      2 # see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
      3 # for examples
      4 echo .bashrc HEAD $-
      5 
      6 # If not running interactively, don't do anything
      7 case $- in
      8     *i*) ;;
      9       *) return;;
     10 esac
     11 
    ...
    122 
    123 export LC_ALL=en_US.UTF-8
    124 
    125 vip(){ vim $HOME/.bashrc ; }
    126 ini(){ source $HOME/.bashrc ; }
    127 
    128 export OPTICKS_HOME=$HOME/opticks
    129 export LOCAL_BASE=/usr/local
    130 opticks-(){ . $OPTICKS_HOME/opticks.bash && opticks-env $* && opticks-export ; }
    131 
    132 export PYTHONPATH=$HOME
    133 export PATH=$LOCAL_BASE/opticks/lib:$OPTICKS_HOME/bin:$OPTICKS_HOME/ana:$PATH
    134 
    135 
    136 o(){ opticks- ; cd $(opticks-home) ; hg st ; }
    137 on(){ cd $OPTICKS_HOME/notes/issues ; }
    138 t(){ type $* ; }
    139 
    140 opticks-
    141 
    142 echo .bashrc TAIL $-

.bash_profile::

      001 echo .bash_profile HEAD $-
        2 source ~/.bashrc
        3 echo .bash_profile TAIL $-
        4 

::

    blyth@blyth-VirtualBox:~$ t ts
    ts is a function
    ts () 
    { 
        LV=$1 tboolean.sh ${@:2}
    }


The default Ubuntu .bashrc early exits when invoked via "/bin/bash -l" but not with "/bin/bash -i"

With "-l" in tboolean.sh the .bash_profile is called, in which I source .bashrc but it early exits:: 


    blyth@blyth-VirtualBox:~$ ts box
    .bash_profile HEAD hB
    .bashrc HEAD hB
    .bash_profile TAIL hB
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth =================
    /home/blyth/opticks/bin/tboolean.sh: line 55: tboolean-: command not found
    tboolean-lv
    /home/blyth/opticks/bin/tboolean.sh: line 59: tboolean-lv: command not found
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth ============ RC 127 =======


With "-i" in tboolean.sh .bash_profile is not called::

    blyth@blyth-VirtualBox:~$ ts box
    .bashrc HEAD himBH
    .bashrc TAIL himBH
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth =================
    tboolean-lv
    === tboolean-lv : tboolean-box
    Traceback (most recent call last):
      File "<stdin>", line 3, in <module>
      File "opticks/ana/main.py", line 4, in <module>
        import numpy as np
    ImportError: No module named numpy
    === tboolean-box : testconfig

    tboolean-info
    ==================


    BASH_VERSION         : 4.4.19(1)-release
    TESTNAME             : tboolean-box
    TESTCONFIG           : 
    TORCHCONFIG          : 

    tboolean-testname    : tboolean-box
    tboolean-testconfig  : 
    tboolean-torchconfig : type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500


    === tboolean-- : no testconfig : try tboolean-box-
    === tboolean-lv : tboolean-box RC 255
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth ============ RC 255 =======
    blyth@blyth-VirtualBox:~$ 


With just "/bin/bash" in tboolean.sh::

    blyth@blyth-VirtualBox:~$ ts box
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth =================
    /home/blyth/opticks/bin/tboolean.sh: line 55: tboolean-: command not found
    tboolean-lv
    /home/blyth/opticks/bin/tboolean.sh: line 59: tboolean-lv: command not found
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth ============ RC 127 =======
    blyth@blyth-VirtualBox:~$ 


On ssh in .bash_profile called
--------------------------------

::

    [blyth@localhost docs]$ ssh V
    ...
    Last login: Thu Jul  4 10:12:31 2019 from 10.0.2.2
    .bash_profile HEAD himBH
    .bashrc HEAD himBH
    .bashrc TAIL himBH
    .bash_profile TAIL himBH
    blyth@blyth-VirtualBox:~$ 


Hmm whats the best way to handle such system diffs ?
---------------------------------------------------------

1. introduce an .opticks_profile with the Opticks setup
2. instruct users to source that from appropriate places for each distro
   such that scripts which use "/bin/bash -l" will define the bash functions

From:

* http://mywiki.wooledge.org/DotFiles



man bash : .opticks_profile or .opticksrc ?
---------------------------------------------------

* .opticks_profile makes more sense as the funcs are needed non-interactively


When  bash  is  invoked  as  an  interactive login shell, or as a
non-interactive shell with the --login option, it first reads and executes
commands from the file /etc/profile, if that file exists.  After reading that
file, it looks for ~/.bash_profile, ~/.bash_login, and ~/.profile, in that
order, and reads and executes commands from the first one that exists and is
readable.  The --noprofile option may be used when the shell is started to
inhibit this behavior.

When an interactive shell that is not a login shell is started, bash reads and
executes commands from ~/.bashrc, if that file exists.  This may be inhibited
by using the --norc option.  The --rcfile file option will force bash to read
and  execute  commands from file instead of ~/.bashrc.



So change the dot files
----------------------------

.bash_profile::

    printf "%s %40s %s %10s %s %s \n" $0 $BASH_SOURCE HEAD $- OPTICKS_KEY $OPTICKS_KEY
    source ~/.bashrc
    printf "%s %40s %s %10s %s %s \n" $0 $BASH_SOURCE TAIL $- OPTICKS_KEY $OPTICKS_KEY

.bashrc::

    # ~/.bashrc: executed by bash(1) for non-login shells. 
    # see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
    # for examples
    printf "%s %40s %s %10s %s %s \n" $0 $BASH_SOURCE HEAD $- OPTICKS_KEY $OPTICKS_KEY
    
    vip(){ vim ~/.bash_profile ~/.bashrc ~/.opticks_setup ; }
    ini(){ source ~/.bashrc ; }
    source ~/.opticks_setup      # NB prior to below early exit "return"

    # If not running interactively, don't do anything
    case $- in
        *i*) ;;
          *) return;;
    esac

    ... 

    printf "%s %40s %s %10s %s %s \n" $0 $BASH_SOURCE TAIL $- OPTICKS_KEY $OPTICKS_KEY 


.opticks_setup::

    printf "%s %40s %s %10s %s %s \n" $0 $BASH_SOURCE HEAD $- OPTICKS_KEY $OPTICKS_KEY
        
    opticks-setup-notes(){ cat << EOU
    Opticks Recommended Bash Setup
    ==================================

    * ~/.bash_profile should source ~/.bashrc
    * ~/.bashrc should source ~/.opticks_setup PRIOR to any early exits in the script
        
    Using this approach succeeds to setup the opticks bash functions
    and exports with either "bash -l" or "bash -i" which aims to make
    it immune to Linux distro and macOS differing treatments of
    ~/.bash_profile and ~/.bashrc and when to invoke those.

    Also scripts using shebang line "#!/bin/bash -l" should have
    the bash functions.

    EOU
    }


    export LC_ALL=en_US.UTF-8

    export OPTICKS_HOME=$HOME/opticks
    export LOCAL_BASE=/usr/local
    opticks-(){ . $OPTICKS_HOME/opticks.bash && opticks-env $* && opticks-export ; }

    export PYTHONPATH=$HOME
    export PATH=$LOCAL_BASE/opticks/lib:$OPTICKS_HOME/bin:$OPTICKS_HOME/ana:$PATH


    o(){ opticks- ; cd $(opticks-home) ; hg st ; }
    on(){ cd $OPTICKS_HOME/notes/issues ; }
    t(){ type $* ; }
    v(){ vi $(which $1) ; }

    opticks-
    opticks-tboolean-shortcuts   # ts, tv, tv4, ta

    printf "%s %40s %s %10s %s %s \n" $0 $BASH_SOURCE TAIL $- OPTICKS_KEY $OPTICKS_KEY




::

    blyth@blyth-VirtualBox:~$ bash -l
    bash                /home/blyth/.bash_profile HEAD      himBH OPTICKS_KEY  
    bash                      /home/blyth/.bashrc HEAD      himBH OPTICKS_KEY  
    bash               /home/blyth/.opticks_setup HEAD      himBH OPTICKS_KEY  
    bash               /home/blyth/.opticks_setup TAIL      himBH OPTICKS_KEY  
    bash                      /home/blyth/.bashrc TAIL      himBH OPTICKS_KEY  
    bash                /home/blyth/.bash_profile TAIL      himBH OPTICKS_KEY  
    blyth@blyth-VirtualBox:~$ 
    blyth@blyth-VirtualBox:~$ 
    blyth@blyth-VirtualBox:~$ bash -i 
    bash                      /home/blyth/.bashrc HEAD      himBH OPTICKS_KEY  
    bash               /home/blyth/.opticks_setup HEAD      himBH OPTICKS_KEY  
    bash               /home/blyth/.opticks_setup TAIL      himBH OPTICKS_KEY  
    bash                      /home/blyth/.bashrc TAIL      himBH OPTICKS_KEY  
    blyth@blyth-VirtualBox:~$ 
    blyth@blyth-VirtualBox:~$ 



Now with "/bin/bash -l" in tboolean.sh get as far as expect, after "sudo apt install python-numpy"

* notice the early exit that prevents reaching the TAIL of .bashrc 

::

    blyth@blyth-VirtualBox:~$ ts box
    /bin/bash                /home/blyth/.bash_profile HEAD         hB OPTICKS_KEY  
    /bin/bash                      /home/blyth/.bashrc HEAD         hB OPTICKS_KEY  
    /bin/bash               /home/blyth/.opticks_setup HEAD         hB OPTICKS_KEY  
    /bin/bash               /home/blyth/.opticks_setup TAIL         hB OPTICKS_KEY  
    /bin/bash                /home/blyth/.bash_profile TAIL         hB OPTICKS_KEY  
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth =================
    tboolean-lv
    === tboolean-lv : tboolean-box
    args: 
    Traceback (most recent call last):
      File "<stdin>", line 9, in <module>
      File "opticks/ana/main.py", line 292, in opticks_main
        opticks_environment()
      File "opticks/ana/env.py", line 21, in opticks_environment
        env = OpticksEnv()
      File "opticks/ana/env.py", line 109, in __init__
        self.direct_init()
      File "opticks/ana/env.py", line 121, in direct_init
        assert os.environ.has_key("OPTICKS_KEY"), "OPTICKS_KEY envvar is required"
    AssertionError: OPTICKS_KEY envvar is required
    === tboolean-box : testconfig

    tboolean-info
    ==================


    BASH_VERSION         : 4.4.19(1)-release
    TESTNAME             : tboolean-box
    TESTCONFIG           : 
    TORCHCONFIG          : 

    tboolean-testname    : tboolean-box
    tboolean-testconfig  : 
    tboolean-torchconfig : type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500


    === tboolean-- : no testconfig : try tboolean-box-
    === tboolean-lv : tboolean-box RC 255
    ====== /home/blyth/opticks/bin/tboolean.sh ====== PWD /home/blyth ============ RC 255 =======
    blyth@blyth-VirtualBox:~$ 

