owl-vi(){       vi $BASH_SOURCE ; }
owl-usage(){ cat << EOU

OWL : A Node Graph "Wrapper" Library for OptiX 7
=================================================

* https://github.com/owl-project/owl
* https://owl-project.github.io
* ``owlBuildSBT(context)``

Introducing OWL: A Node Graph Abstraction Layer on top of OptiX 7
-------------------------------------------------------------------

* https://ingowald.blog/2020/11/08/introducing-owl-a-node-graph-abstraction-layer-on-top-of-optix-7/


EOU
}
owl-env(){ olocal- ; }
owl-dir(){ echo $(opticks-prefix)/externals/owl ; }
owl-cd(){  cd $(owl-dir) ; }
owl-url(){ echo https://github.com/owl-project/owl ; }
owl-get()
{
    local dir=$(dirname $(owl-dir)) &&  mkdir -p $dir && cd $dir
    [ ! -d owl ] && git clone $(owl-url) 
}

