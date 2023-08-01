#!/bin/bash -l 
usage(){ cat << EOU 
rsync_put.sh 
===================

When the GFW blocks bitbucket and github "put" from remote using rsync 

Usage::

   o
   ./bin/rsync_put.sh




* TODO : add cmd that just syncs working copy 


rsync options
----------------

-z, --compress   compress file data during the transfer
-a, --archive    archive mode; equals -rlptgoD (no -H,-A,-X)   
-r, --recursive  recurse into directories
-v, --verbose    increase verbosity
    --delete     delete extraneous files from dest dirs

* :google:`rsync git repository`

* https://tylercipriani.com/blog/2020/09/22/migrating-git-data-with-rsync/

::

    "The moral of the story here is to never omit --delete from rsync if you’re
    trying to keep repos in sync."


git bundle
-----------

Another approach see "git help bundle"


observation 1
----------------

A do almost nothing put takes about 20s::

    Tue Aug  1 15:44:00 BST 2023
    sending incremental file list
    .git/
    deleting bin/rsync_put_repo.sh
    bin/
    bin/rsync_put.sh

    sent 511,511 bytes  received 893 bytes  27,697.51 bytes/sec
    total size is 211,056,404  speedup is 411.89
    Tue Aug  1 15:44:19 BST 2023
    epsilon:opticks blyth$ 

observation 2
----------------

Some files are missing at other end::

    N[blyth@localhost junotop]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes not staged for commit:
      (use "git add/rm <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        deleted:    analytic/gdml.py
        deleted:    qudarap/tests/qstateTest.cc
        deleted:    qudarap/tests/qstateTest.sh

    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        bin/rsync_put.sh

    no changes added to commit (use "git add" and/or "git commit -a")
    N[blyth@localhost opticks]$ l qudarap/tests/qstateTest.cc
    ls: cannot access qudarap/tests/qstateTest.cc: No such file or directory
    N[blyth@localhost opticks]$ l qudarap/tests/qstateTest.sh
    ls: cannot access qudarap/tests/qstateTest.sh: No such file or directory
    N[blyth@localhost opticks]$ pwd
    /data/blyth/junotop/opticks
    N[blyth@localhost opticks]$ l analytic/gdml.py
    ls: cannot access analytic/gdml.py: No such file or directory
    N[blyth@localhost opticks]$ 
     
What is special about these three::

    epsilon:opticks blyth$ l qudarap/tests/qstateTest.cc qudarap/tests/qstateTest.sh analytic/gdml.py 
    112 -rwxr-xr-x  1 blyth  staff  55734 Aug  1 15:48 analytic/gdml.py
      8 -rw-r--r--  1 blyth  staff    366 Oct  1  2022 qudarap/tests/qstateTest.cc
      8 -rwxr-xr-x  1 blyth  staff    326 Mar 26  2022 qudarap/tests/qstateTest.sh
    epsilon:opticks blyth$ 


Looks like some case irregularities::

    epsilon:tests blyth$ l qstate* QState*
    8 -rw-r--r--  1 blyth  staff  366 Aug  1 15:54 QStateTest.cc
    8 -rwxr-xr-x  1 blyth  staff  326 Aug  1 15:52 qstateTest.sh
    epsilon:tests blyth$ 

After fiddling and sync again::

    N[blyth@localhost opticks]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes not staged for commit:
      (use "git add/rm <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        deleted:    qudarap/tests/qstateTest.cc

    no changes added to commit (use "git add" and/or "git commit -a")
    N[blyth@localhost opticks]$ 




EOU
}

defarg="info_all"
arg=${1:-$defarg}

dstfold=/data/blyth/junotop
#dstfold=/tmp/blyth/junotop

src=$HOME/opticks/         ## for rsync a trailing slash on source directory means copy contents of this directory
dst=${dstfold}/opticks
to=P:$dst

vars="BASH_SOURCE defarg arg dstfold src dst to"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/all}" != "$arg" ]; then
   date 
   rsync -zarv --delete "$from" "$to"
   date
   [ $? -ne 0 ] && echo $BASH_SOURCE : all error && exit 1 
fi 


if [ "${arg/wc}" != "$arg" ]; then

   read -p "$BASH_SOURCE : CAUTION UNTESTED : this will delete .git 160MB at the other end, enter YES to continue " answer

   if [ "$answer" == "YES" ]; then  

       date 
       rsync -zarv --exclude .git/ --delete "$from" "$to"
       date
       [ $? -ne 0 ] && echo $BASH_SOURCE : wc error && exit 2 
   else
        echo $BASH_SOURCE : skip 
   fi 


fi 




