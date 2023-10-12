#!/bin/bash -l 
usage(){ cat << EOU | sed 's/^/# /' 
addtag.sh
==========

NB RUNNING THIS SCRIPT ONLY EMITS COMMANDS TO STDOUT : PIPE TO SHELL TO RUN THEM 

Workflow for adding Opticks tags:

1. edit okconf/OpticksVersionNumber.hh increasing OPTICKS_VERSION_NUMBER
   to correspond to the next intended tag string and add table entry for the next tag 
   to notes/releases-and-versioning.rst::

       cd ~/opticks
       vi okconf/OpticksVersionNumber.hh notes/releases-and-versioning.rst

   For example when starting from "#define OPTICKS_VERSION_NUMBER 16" 
   with last "git tag" v0.1.6 and intended next tag v0.1.7 the OPTICKS_VERSION_NUMBER 
   would simply be incremented to 17.  However after a long period without tags it 
   might be appropriate to jump to a new minor version, changing OPTICKS_VERSION_NUMBER 
   to 20 and tag to v0.2.0 

2. commit changes including okconf/OpticksVersionNumber.hh::

       git commit -m "Prepare to resume tagging with addtag.sh v0.2.0 OPTICKS_VERSION_NUMBER 20" 

3. push code changes to BOTH bitbucket and github::

       git push 
       git push github
      
4. run this tag add and pushing script, check output commands and run if correct::

       cd ~/opticks
       ./addtag.sh           # check tag add and push commands 
       ./addtag.sh | sh      # run those commands
       open https://github.com/simoncblyth/opticks/tags # check web interface

NB this simple script assumes single digit 0-9 major/minor/patch version integers

EOU
}
usage

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
git push github --tags

EOC

exit 0 
