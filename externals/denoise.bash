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

