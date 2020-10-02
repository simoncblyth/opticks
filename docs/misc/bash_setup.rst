Bash Setup
=============


.bash_profile OR .bashrc, macOS and Linux
-------------------------------------------

With most Linux distributions and terminal managers the *.bash_profile* is run
only on login and *.bashrc* is run for every new terminal window, BUT with macOS Terminal.app
the *.bash_profile* is run for every new terminal window.  Thus for compatibility 
the best approach to put setup into *.bashrc* and source it from *.bash_profile* : giving 
the same behaviour on both Linux and macOS.

For background on dotfiles http://mywiki.wooledge.org/DotFiles


Examples .bash_profile
------------------------

Example `~/.bash_profile`:

.. code-block:: sh

    # .bash_profile

    if [ -f ~/.bashrc ]; then                 ## typical setup 
            . ~/.bashrc
    fi



Ubuntu early exit .bashrc
-----------------------------

Some Linux distros (Ubuntu) have a default `.bashrc` which early exits. 
It is necessary to *source ~/.opticks_config* prior to the early exit.  
Example `~/.bashrc`:

.. code-block:: sh

    # .bashrc

    vip(){ vim ~/.bash_profile ~/.bashrc ~/.opticks_config ; } 
    ini(){ source ~/.bashrc ; } 

    source ~/.opticks_config

    ##### below from default Ubuntu .bashrc early exits if bash is not invoked with -i option 

    # If not running interactively, don't do anything
    case $- in
        *i*) ;;
          *) return;;
    esac


For notes about this see `notes/issues/ubuntu-bash-login-shell-differences.rst`



