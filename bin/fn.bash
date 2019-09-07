#!/usr/bin/env bash
##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##


fn-env(){ echo -n ; }
fn-vi(){ vi $BASH_SOURCE ; }

fn-usage(){ cat << EOU

This attempts to reproduce an issue with macOS bash, 
seen with tboolean-box 

fn.bash
   analog for tboolean.bash

fn.sh
   analog for tboolean.sh


Currently it fails to reproduce the problem of TESTCONFIG not being defined::

    epsilon:~ blyth$ LV=box fn.sh
    fn-lv

        BASH_VERSION : 5.0.7(1)-release
        FUNCNAME     : fn--
        TESTNAME     : fn-box
        TESTCONFIG   : 42 

    === fn-lv : fn-box RC 0
    /Users/blyth/opticks/bin/fn.sh rc 0
    epsilon:~ blyth$ 
    epsilon:~ blyth$ LV=sox fn.sh
    fn-lv

        BASH_VERSION : 5.0.7(1)-release
        FUNCNAME     : fn--
        TESTNAME     : fn-sox
        TESTCONFIG   : 142 

    === fn-lv : fn-sox RC 0
    /Users/blyth/opticks/bin/fn.sh rc 0

Under cmake also::

     om-test -V


EOU
}

fn-funcname(){ 
   local funcname="fn-$LV"
   echo $funcname  
}

fn-lv(){
   local msg="=== $FUNCNAME :"
   local funcname=$(fn-funcname)
   $funcname $*   
   RC=$?
   echo $msg $funcname RC $RC
   return $RC
}

fn--()
{
    cat << EOF

    BASH_VERSION : $BASH_VERSION
    FUNCNAME     : $FUNCNAME
    TESTNAME     : $TESTNAME
    TESTCONFIG   : $TESTCONFIG 

EOF
}

fn-box(){    TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) fn-- $* ; }
fn-box-(){ $FUNCNAME- | python ; }
fn-box--(){ cat << EOP
print(42)
EOP
}


fn-sox(){    TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) fn-- $* ; }
fn-sox-(){ $FUNCNAME- | python ; }
fn-sox--(){ cat << EOP
print(142)
EOP
}



