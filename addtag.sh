#!/bin/bash
usage(){ cat << EOU | sed 's/^/# /'
addtag.sh
==========

NB RUNNING THIS SCRIPT ONLY EMITS COMMANDS TO STDOUT : PIPE TO SHELL TO RUN THEM

Workflow for adding Opticks tags:

0. check if now is an appropriate time to tag, by running tests::

   oo                       ## update build
   opticks-setup-generate   ## if have changed the setup bashrc
   opticks-t

1. edit okconf/OpticksVersionNumber.hh increasing OPTICKS_VERSION_NUMBER
   to correspond to the next intended tag string and add table entry for the next tag
   to notes/releases-and-versioning.rst::

       cd ~/opticks
       git lg -n10
       vi okconf/OpticksVersionNumber.hh notes/releases-and-versioning.rst

   For example when starting from "#define OPTICKS_VERSION_NUMBER 16"
   with last "git tag" v0.1.6 and intended next tag v0.1.7 the OPTICKS_VERSION_NUMBER
   would simply be incremented to 17.  However after a long period without tags it
   might be appropriate to jump to a new minor version, changing OPTICKS_VERSION_NUMBER
   to 20 and tag to v0.2.0

2. commit changes including okconf/OpticksVersionNumber.hh::

       git status
       git add okconf/OpticksVersionNumber.hh notes/releases-and-versioning.rst
       git commit -m "Prepare to ./addtag.sh $vntag OPTICKS_VERSION_NUMBER $opticks_version_number "

3. push code changes to BOTH bitbucket and github::

       cd ~/opticks
       git push

       # if not done already add "bitbucket" and "github" remotes
       git remote -v
       git remote add bitbucket git@bitbucket.org:simoncblyth/opticks.git
       git remote add github git@github.com:simoncblyth/opticks.git
       git remote -v

       # to change remote origin while proxying to others stop working
       git remote set-url origin  git@github.com:simoncblyth/opticks.git
       git remote set-url origin  git@bitbucket.org:simoncblyth/opticks.git
       git remote set-url origin  git@gitlab.com:simoncblyth/opticks.git

       git push              ## to whichever is currently origin
       git push github
       git push bitbucket


4. run this tag add and pushing script, check output commands and run if correct::

       cd ~/opticks
       ./addtag.sh           # check tag add and push commands
       ./addtag.sh | sh      # run those commands
       open https://github.com/simoncblyth/opticks/tags # check web interface

5. create distribution tarball for the release::

       okdist-;okdist--

6. scp the okdist tarball to O and deploy to eg /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/
   and update the Opticks-vLatest link::

       okdist-;okdist-deploy-to-cvmfs

   OR do that manually replacing the appropriate version in the below::

       A> scp /data1/blyth/local/opticks_Debug/Opticks-v0.3.2.tar O:
       A> ssh O
       O> ./ok_deploy_to_cvmfs.sh Opticks-v0.3.2.tar   ## cvmfs details in hcvmfs-


7. [JUNOSW+Opticks release] After the "day name" automatic ~/.gitlab-ci.yml
   deployment to CVMFS of the OJ tarball has been checked, add a dated reference release
   for the exact same OJ tarball by:

   *  SSH into the OJ machine "ssh O" and invoke ./oj_reference_deploy_to_cvmfs.sh


NB this simple script assumes single digit 0-9 major/minor/patch version integers

EOU
}

sdir=$(cd $(dirname $BASH_SOURCE) && pwd)
vctag=$(git tag | sed -n '$p') # same a tail -1 : pick the last listed tag
ctag=${vctag:1}
ctag_major=$(echo $ctag | cut -f 1 -d .)
ctag_minor=$(echo $ctag | cut -f 2 -d .)
ctag_patch=$(echo $ctag | cut -f 3 -d .)
## assuming simple dotted triplet like v0.1.6

if [ $ctag_major -eq 0 ]; then
    ctag_num=${ctag_minor}${ctag_patch}
else
    ctag_num=${ctag_major}${ctag_minor}${ctag_patch}
fi
ctag_note="ctag : current tag obtained from last listed git tag"


statline=$(cd $sdir && git status --porcelain | wc -l )
opticks_version_number=$(cat $sdir/okconf/OpticksVersionNumber.hh | sed -n '/^#define OPTICKS_VERSION_NUMBER/p' | cut -f 3 -d " ")

if [ ${#opticks_version_number} -eq 3 ]; then
   ntag_major=${opticks_version_number:0:1}
   ntag_minor=${opticks_version_number:1:1}
   ntag_patch=${opticks_version_number:2:1}
   ntag_num=${ntag_major}${ntag_minor}${ntag_patch}
elif [ ${#opticks_version_number} -eq 2 ]; then
   ntag_major=0
   ntag_minor=${opticks_version_number:0:1}
   ntag_patch=${opticks_version_number:1:1}
   ntag_num=${ntag_minor}${ntag_patch}
fi
ntag=${ntag_major}.${ntag_minor}.${ntag_patch}
vntag=v$ntag

usage


ntag_note="ntag : next tag obtained from okconf/OpticksVersionNumber.hh"

vars="BASH_SOURCE sdir"
vars="$vars ctag_note vctag ctag ctag_major ctag_minor ctag_patch ctag_num"
vars="$vars opticks_version_number"
vars="$vars ntag_note vntag ntag ntag_major ntag_minor ntag_patch ntag_num"
vars="$vars statline"

for var in $vars ; do printf "# %25s : %s\n" "$var" "${!var}" ; done
echo "#"




err="# $BASH_SOURCE : ERROR "
st=0
[ $statline -ne 0 ]    && echo $err : \"git status --porcelain\" must return zero lines to addtag : not $statline        && st=1
[ $opticks_version_number -le $ctag_num ] && echo $err : must increase okconf/OpticksVersionNumber.hh OPTICKS_VERSION_NUMBER before addtag && st=2
[ $st -ne 0 ] && echo $err : ERROR CONDITIONS NOTED ABOVE PREVENT TAGGING BY COMMENTING THE BELOW COMMANDS
[ $st -eq 0 ] && pfx="" || pfx="# "


cat << EOC | sed "s/^/$pfx/"

git tag -a $vntag -m "OPTICKS_VERSION_NUMBER ${ntag_num}"
git push --tags
EOC

origin=$(git remote get-url origin)
if [ "${origin/github}" != "$origin" -a -z "$SKIP_BITBUCKET" ]; then
   echo ${pfx}git push bitbucket --tags  \# as origin looks to be github
elif [ "${origin/bitbucket}" != "$origin" -a -z "$SKIP_GITHUB" ]; then
   echo ${pfx}git push github --tags  \# as origin looks to be bitbucket
else
   echo \#\# failed to identify origin $origin or are skipping SKIP_BITBUCKET [$SKIP_BITBUCKET] SKIP_GITHUB [$SKIP_GITHUB]
fi
echo

exit 0
