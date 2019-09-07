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


#OKTest --size 1920,1080,1 --position 100,100 
#OKTest --size 1920,1080,1 --position 100,100 --gltf 3

#op.sh --j1707 --gltf 3 --scintillation --save --compute --timemax 400 --animtimemax 400 
#op.sh --j1707 --gltf 3 --scintillation --load --timemax 400 --animtimemax 400

#op.sh --j1808 --gltf 3 --scintillation --save --compute --timemax 400 --animtimemax 400 
#op.sh --j1808 --gltf 3 --scintillation --load --timemax 400 --animtimemax 400
# 1808 not in legacy workflow ? 
#

OKTest --envkey --xanalytic --geocenter
#OKG4Test --envkey --xanalytic --geocenter


 
