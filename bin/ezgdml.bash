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

ezgdml-source(){ echo $BASH_SOURCE ; }
ezgdml-vi(){ vi $(ezgdml-source)  ; }
ezgdml-env(){  olocal- ; opticks- ; geocache- ;  }
ezgdml-usage(){ cat << EOU

EOU
}

ezgdml-path(){ echo g4codegen/tests/x016.gdml ; }

ezgdml--()
{
   geocache-kcd
   ipython -i $(which ezgdml.py) -- $(ezgdml-path)
}



