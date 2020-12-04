owl-vi(){       vi $BASH_SOURCE ; }
owl-usage(){ cat << EOU

OWL : A Node Graph "Wrapper" Library for OptiX 7
=================================================

* https://github.com/owl-project/owl

EOU
}
owl-env(){ olocal- ; }
owl-dir(){ echo /tmp/$USER/opticks/owl ; }
owl-cd(){  cd $(owl-dir) ; }
owl-url(){ echo https://github.com/owl-project/owl ; }
owl-get()
{
    local dir=$(dirname $(owl-dir)) &&  mkdir -p $dir && cd $dir
    [ ! -d owl ] && git clone $(owl-url) 
}

