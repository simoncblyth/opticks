#!/bin/bash 
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


runner-usage(){ cat << \EOU

CTest Runner [NOT USED]
==========================

**Unfortunately find inconsistency between CMake versions regarding TARGET_FILE 
so have stopped using this test runner approach.**

Instead remote session detection is done in SSys::IsRemoteSession
and ggeoview-/App::isCompute flips to ON even without "--compute" option when
a remote session is detected.




Currently only the GGeoView tests are run using this
test runner as they need the "--compute" option to be 
appended when running remotely to avoid failures
from remote X11 window creation.

See ggeoview/tests/CMakeLists.txt for configuration 
of this runner::

    add_test(NAME ${name}.${TGT} COMMAND OpticksCTestRunner.sh --config $<CONFIGURATION> --exe $<TARGET_FILE:${TGT}>)

An example invokation::

   /home/simonblyth/opticks/bin/OpticksCTestRunner.sh "--config" "Debug" "--exe" "/home/simonblyth/local/opticks/build/ggeoview/tests/OTracerTest"





EOU
}


CONFIG=""
EXE=""
RARGS=""

runner-argparse()
{
    while [[ $# -gt 1 ]]
    do
        case $1 in
           --config) CONFIG=$2 ; shift ;;
              --exe) EXE=$2    ; shift ;;
      --remote-args) RARGS=$2    ; shift ;;
                  *) echo "unknown option $1" ;;
        esac
        shift   # past the value  
    done
}
runner-remote-session()
{
   [ -n "$SSH_CLIENT" -o -n "$SSH_TTY" ] && echo YES || echo NO
}
runner-dump()
{
   cat << EOD

   CONFIG     $CONFIG
   EXE        $EXE 
   RARGS      $RARGS

   SSH_CLIENT  \"$SSH_CLIENT\" 
   SSH_TTY     \"$SSH_TTY\"
   REMOTE      \"$REMOTE\" 

   RUNLINE     \"$RUNLINE\"

EOD
}

runner-argparse $*

REMOTE=$(runner-remote-session)

OPTIONS="" 
[ "$REMOTE" == "YES" ] && OPTIONS="$OPTIONS $RARGS"  

RUNLINE="$EXE $OPTIONS"

runner-dump

echo $RUNLINE
eval $RUNLINE
RC=$?
echo RC $RC
exit $RC

