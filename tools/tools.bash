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
