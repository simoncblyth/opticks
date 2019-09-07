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

boost-(){            . $(opticks-home)/externals/boost.bash             && boost-env $* ; }
ocmake-(){           . $(opticks-home)/externals/ocmake.bash            && ocmake-env $* ; }
glm-(){              . $(opticks-home)/externals/glm.bash               && glm-env $* ; }
plog-(){             . $(opticks-home)/externals/plog.bash              && plog-env $* ; }
gleq-(){             . $(opticks-home)/externals/gleq.bash              && gleq-env $* ; }
glfw-(){             . $(opticks-home)/externals/glfw.bash              && glfw-env $* ; }
glew-(){             . $(opticks-home)/externals/glew.bash              && glew-env $* ; }
imgui-(){            . $(opticks-home)/externals/imgui.bash             && imgui-env $* ; }
assimp-(){           . $(opticks-home)/externals/assimp.bash            && assimp-env $* ; }
openmesh-(){         . $(opticks-home)/externals/openmesh.bash          && openmesh-env $* ; }
cuda-(){             . $(opticks-home)/externals/cuda.bash              && cuda-env $* ; }
cudamac-(){          . $(opticks-home)/externals/cudamac.bash           && cudamac-env $* ; }
cudalin-(){          . $(opticks-home)/externals/cudalin.bash           && cudalin-env $* ; }
cu-(){               . $(opticks-home)/externals/cuda.bash              && cuda-env $* ; }
thrust-(){           . $(opticks-home)/externals/thrust.bash            && thrust-env $* ; }
optix-(){            . $(opticks-home)/externals/optix.bash             && optix-env $* ; }
optixnote-(){        . $(opticks-home)/externals/optixnote.bash         && optixnote-env $* ; }
xercesc-(){          . $(opticks-home)/externals/xercesc.bash           && xercesc-env $* ; }
g4-(){               . $(opticks-home)/externals/g4.bash                && g4-env $* ; }
zmq-(){              . $(opticks-home)/externals/zmq.bash               && zmq-env $* ; }
asiozmq-(){          . $(opticks-home)/externals/asiozmq.bash           && asiozmq-env $* ; }
opticksdata-(){      . $(opticks-home)/externals/opticksdata.bash       && opticksdata-env $* ; }

oimplicitmesher-(){  . $(opticks-home)/externals/oimplicitmesher.bash   && oimplicitmesher-env $* ; }
odcs-(){             . $(opticks-home)/externals/odcs.bash              && odcs-env $* ; }
oyoctogl-(){         . $(opticks-home)/externals/oyoctogl.bash          && oyoctogl-env $* ; }
ocsgbsp-(){          . $(opticks-home)/externals/ocsgbsp.bash           && ocsgbsp-env $* ; }
oof-(){              . $(opticks-home)/externals/oof.bash               && oof-env $* ; }
bcm-(){              . $(opticks-home)/externals/bcm.bash               && bcm-env $* ; }

g4dev-(){            . $(opticks-home)/externals/g4dev.bash             && g4dev-env $* ; }
g4dae-(){            . $(opticks-home)/externals/g4dae.bash             && g4dae-env $* ; }

