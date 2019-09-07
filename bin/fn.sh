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


notes(){ cat << EON


* https://stackoverflow.com/questions/4303128/how-to-use-multiple-arguments-for-awk-with-a-shebang-i-e

Normal approach of using "bash -l" to source the 
login scripts doesnt work on Linux together with shebang
(it does work on macOS):: 

     #!/usr/bin/env bash -l

So in the below find OPTICKS_HOME from the path to this script.
But this is not a practical general solution as users exporting 
OPTICKS_KEY in .bashrc to pick a geometry is expected usage.

EON
}


export OPTICKS_HOME=$(dirname $(dirname $0))
opticks-(){ . $OPTICKS_HOME/opticks.bash && opticks-env $* ; } 
opticks-

fn-

cmd="fn-lv $*"

echo $cmd
eval $cmd
rc=$?

echo $0 rc $rc
exit $rc

