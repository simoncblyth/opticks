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

denoise-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(denoise-src)} ; }
denoise-vi(){       vi $(denoise-source) ; }
denoise-env(){      olocal- ; }
denoise-usage(){ cat << EOU



* :google:`noise2noise`

* https://arxiv.org/pdf/1803.04189.pdf


* :google:`Rendered Image Denoising using Autoencoders`

* https://www.mahmoudhesham.net/blog/post/using-autoencoder-neural-network-denoise-renders



EOU
}

