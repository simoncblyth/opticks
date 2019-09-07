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

tools-src(){      echo tools/tools.bash ; }
tools-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(tools-src)} ; }
tools-vi(){       vi $(tools-source) ; }

tools-usage(){ cat << EOU


EOU
}

tools-env(){     
    olocal- 
}


tools-sdir(){ echo $(opticks-home)/tools; }
tools-c(){    cd $(tools-sdir)/$1 ; }
tools-cd(){   cd $(tools-sdir)/$1 ; }


tools-i()
{
   tools-c

   /usr/bin/python -i standalone.py  

}
