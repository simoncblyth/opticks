#!/bin/bash -l 
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


tboolean-sh-notes(){ cat << EON
/**
tboolean.sh
==============

NB notice the "/bin/bash -l"
------------------------------

The "-l" means that login environment scripts 
including .bashrc are invoked before these 
functions run.  For the below "tboolean-" to 
be defined .bashrc needs to have defined and run
the "opticks-" precursor function. 


Direct Approach
-----------------

* setup executable with the new o.sh using --okg4 option
  to select OKG4Test 
* test geometry is based off the direct geocache
  that is pointed to by OPTICKS_KEY envvar 
* material/surface names need to match those in the 
  base geometry 

Legacy Approach
-----------------

* executables and environment are setup by op.sh 
  using --okg4 option to select OKG4Test 

Minimizing this
-----------------

Bash on macOS dislikes too many layers of bash scripts
and functions (its failing to pass in a TESTCONFIG),
so moved most of the argument setup into tboolean-
leaving just directory and environment setup here. 

Formerly invoked geocache-key-export from here 
but choice of base geometry belongs in users bashrc not here. 

So need to instruct users to geocache-recreate 
and set OPTICKS_KEY in .bashrc : which is good as it shows them how 
easy (in principal) it now is to translate a geometry.

**/
EON
}

echo ====== $0 $arg $* ====== PWD $PWD =================

tboolean-
cmd="tboolean-lv $*"

echo $cmd
eval $cmd
rc=$?

echo ====== $0 $arg $* ====== PWD $PWD ============ RC $rc =======

exit $rc
