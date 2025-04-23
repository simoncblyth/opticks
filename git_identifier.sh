#!/bin/bash 
git_identifier_usage(){ cat << EOU
Provide an identifier string for a git repository
===================================================

This script expects to be kept at the top level of a git repository. 

::

   ./git_identifier.sh 
   ./git_identifier.sh NOTE

    A[blyth@localhost j]$ ./git_identifier.sh
    j_notag_9685c51_20250423103025

    A[blyth@localhost j]$ ./git_identifier.sh NOTE
    uncommitted changes in working copy : so include timestamp in identifier


    A[blyth@localhost opticks]$ ./git_identifier.sh 
    opticks_v0.3.8_8ef31c325_20250423103326

    A[blyth@localhost opticks]$ ./git_identifier.sh NOTE
    uncommitted changes in working copy : so include timestamp in identifier

EOU
}


git_identifier()
{
   local arg=$1
   local ltag=$(git tag | tail -1)
   local ltaghash

   if [ "$ltag" == "" ]; then
       ltag="notag"
       ltaghash="placeholder"
   else
       ltaghash=$(git rev-parse --short ${ltag}^{})
   fi 
   local headhash=$(git rev-parse --short HEAD) 
   local porcelain=$(git status --porcelain)
   local timestamp=$(date +"%Y%m%d%H%M%S")

   local pfx=$(basename $PWD)
   local identifier
   local note 

   if [ "$porcelain" != "" ]; then 
        note="uncommitted changes in working copy : so include timestamp in identifier"
        identifier=${pfx}_${ltag}_${headhash}_${timestamp}
   else
        if [ "$ltaghash" == "$headhash" ]; then
            note="no uncommited changes and HEAD repo commit matches the last tag hash : so identify with last tag" 
            identifier=${pfx}_$ltag 
        else
            note="no uncommitted changes but there have been commits since the last tag : so identify with last tag and HEAD commit hash"  
            identifier=${pfx}_${ltag}_${headhash}
        fi 
    fi 

    case $arg in 
      IDENTIFIER) echo $identifier ;; 
      NOTE)       echo $note ;; 
      TIME)       echo $timestamp ;; 
    esac
    return 0
}

cd $(dirname $(realpath $BASH_SOURCE))
arg=${1:-IDENTIFIER}
git_identifier $arg

