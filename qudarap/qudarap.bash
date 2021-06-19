qudarap-source(){   echo $BASH_SOURCE ; }
qudarap-vi(){       vi $(qudarap-source) ; }
qudarap-usage(){ cat << "EOU"
QUDARap
==========

EOU
}

qudarap-env(){      
  olocal-  
}


qudarap-idir(){ echo $(opticks-idir); }
qudarap-bdir(){ echo $(opticks-bdir)/qudarap ; }
qudarap-sdir(){ echo $(opticks-home)/qudarap ; }
qudarap-tdir(){ echo $(opticks-home)/qudarap/tests ; }

qudarap-c(){    cd $(qudarap-sdir)/$1 ; }
qudarap-cd(){   cd $(qudarap-sdir)/$1 ; }
qudarap-scd(){  cd $(qudarap-sdir); }
qudarap-tcd(){  cd $(qudarap-tdir); }
qudarap-bcd(){  cd $(qudarap-bdir); }





