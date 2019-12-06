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

externals-source(){ echo $BASH_SOURCE ; }
externals-vi(){     vi $BASH_SOURCE ; }
externals-dir(){    echo $(dirname $BASH_SOURCE) ; }

boost-(){            . $(externals-dir)/boost.bash             && boost-env $* ; }
ocmake-(){           . $(externals-dir)/ocmake.bash            && ocmake-env $* ; }
glm-(){              . $(externals-dir)/glm.bash               && glm-env $* ; }
plog-(){             . $(externals-dir)/plog.bash              && plog-env $* ; }
gleq-(){             . $(externals-dir)/gleq.bash              && gleq-env $* ; }
glfw-(){             . $(externals-dir)/glfw.bash              && glfw-env $* ; }
glew-(){             . $(externals-dir)/glew.bash              && glew-env $* ; }
imgui-(){            . $(externals-dir)/imgui.bash             && imgui-env $* ; }
assimp-(){           . $(externals-dir)/assimp.bash            && assimp-env $* ; }
openmesh-(){         . $(externals-dir)/openmesh.bash          && openmesh-env $* ; }
cuda-(){             . $(externals-dir)/cuda.bash              && cuda-env $* ; }
cudamac-(){          . $(externals-dir)/cudamac.bash           && cudamac-env $* ; }
cudalin-(){          . $(externals-dir)/cudalin.bash           && cudalin-env $* ; }
cu-(){               . $(externals-dir)/cuda.bash              && cuda-env $* ; }
optix-(){            . $(externals-dir)/optix.bash             && optix-env $* ; }
optix7-(){           . $(externals-dir)/optix7.bash            && optix7-env $* ; }
optix7c-(){          . $(externals-dir)/optix7c.bash           && optix7c-env $* ; }
optixnote-(){        . $(externals-dir)/optixnote.bash         && optixnote-env $* ; }
xercesc-(){          . $(externals-dir)/xercesc.bash           && xercesc-env $* ; }
g4-(){               . $(externals-dir)/g4.bash                && g4-env $* ; }
zmq-(){              . $(externals-dir)/zmq.bash               && zmq-env $* ; }
asiozmq-(){          . $(externals-dir)/asiozmq.bash           && asiozmq-env $* ; }
opticksdata-(){      . $(externals-dir)/opticksdata.bash       && opticksdata-env $* ; }
opticksaux-(){       . $(externals-dir)/opticksaux.bash        && opticksaux-env $* ; }

oimplicitmesher-(){  . $(externals-dir)/oimplicitmesher.bash   && oimplicitmesher-env $* ; }
odcs-(){             . $(externals-dir)/odcs.bash              && odcs-env $* ; }
oyoctogl-(){         . $(externals-dir)/oyoctogl.bash          && oyoctogl-env $* ; }
ocsgbsp-(){          . $(externals-dir)/ocsgbsp.bash           && ocsgbsp-env $* ; }
oof-(){              . $(externals-dir)/oof.bash               && oof-env $* ; }
bcm-(){              . $(externals-dir)/bcm.bash               && bcm-env $* ; }

thrust-(){           . $(externals-dir)/thrust.bash            && thrust-env $* ; }
cub-(){              . $(externals-dir)/cub.bash               && cub-env $* ; }
mgpu-(){             . $(externals-dir)/mgpu.bash              && mgpu-env $* ; }

g4dev-(){            . $(externals-dir)/g4dev.bash             && g4dev-env $* ; }
g4dae-(){            . $(externals-dir)/g4dae.bash             && g4dae-env $* ; }

