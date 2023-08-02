#!/bin/bash -l 
usage(){ cat << EOU 
rsync_put.sh 
===================


When the GFW blocks bitbucket and github "put" from remote using rsync 

Usage::

   ~/opticks/bin/rsync_put.sh


HMM : how dodgy a thing is this ? 

* it means a git binary version is having to deal with the raw 
  repo files provide by another version 



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


observations
----------------

* Doing almost nothing sync takes about 20s
* use of "--delete" is a bit fraught, so this script
  is deliberately not very flexible to avoid surprises


EOU
}

defarg="info_all"
arg=${1:-$defarg}

src=/Users/blyth/opticks/         ## for rsync a trailing slash on source directory means copy contents of this directory
dst=P:/data/blyth/junotop/opticks
cmd="rsync -zarv --delete  $src $dst"
cmdx="rsync -zarv --exclude .git/ --delete $src $dst"


vars="BASH_SOURCE defarg arg src dst cmd cmdx"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/all}" != "$arg" ]; then
   date 
   echo $cmd
   eval $cmd
   [ $? -ne 0 ] && echo $BASH_SOURCE : all error && exit 1 
   date
fi 

if [ "${arg/wc}" != "$arg" ]; then
   read -p "$BASH_SOURCE : CAUTION UNTESTED : this will delete .git 160MB at the other end, enter YES to continue " answer
   if [ "$answer" == "YES" ]; then  

       date 
       echo $cmd
       eval $cmd
       [ $? -ne 0 ] && echo $BASH_SOURCE : wc error && exit 2 
       date
   else
        echo $BASH_SOURCE : skip 
   fi 
fi 



notes(){ cat << EON

Case irregularity::

    epsilon:opticks blyth$ git show HEAD:qudarap/tests/qstateTest.cc
    #include "QState.hh"

    const char* FOLD = "/tmp/QStateTest" ; 

    int main(int argc, char** argv)
    {
        sstate s0 = QState::Make(); 
        std::cout << " s0 " << QState::Desc(s0) << std::endl ; 

        QState::Save(s0, FOLD, "s.npy" ); 

        sstate s1 ; 
        QState::Load(s1, FOLD, "s.npy" ); 
        std::cout << " s1 " << QState::Desc(s1) << std::endl ; 

        return 0 ; 
    }
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ git show HEAD:qudarap/tests/QStateTest.cc
    fatal: Path 'qudarap/tests/QStateTest.cc' exists on disk, but not in 'HEAD'.
    epsilon:opticks blyth$ 

Looks like renamed in file system bit not in git repo, on macOS that 
is not noticed but it is in git.  So on laptop::

    epsilon:tests blyth$ mv QStateTest.cc qstateTest.cc
    epsilon:tests blyth$ git mv qstateTest.cc QStateTest.cc


EON
}

