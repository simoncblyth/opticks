# === func-gen- : graphics/ciexyz/ciexyz fgp graphics/ciexyz/ciexyz.bash fgn ciexyz fgh graphics/ciexyz
ciexyz-src(){      echo graphics/ciexyz/ciexyz.bash ; }
ciexyz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ciexyz-src)} ; }
ciexyz-vi(){       vi $(ciexyz-source) ; }
ciexyz-env(){      elocal- ; }
ciexyz-usage(){ cat << EOU


Simple Analytic Approximations to the CIE XYZ Color Matching Functions

* http://www.ppsloan.org/publications/XYZJCGT.pdf


* http://mathematica.stackexchange.com/questions/57389/convert-spectral-distribution-to-rgb-color

* http://www.cvrl.org


Related

* https://www.fourmilab.ch/documents/specrend/



EOU
}
ciexyz-dir(){ echo $(env-home)/graphics/ciexyz ; }
ciexyz-cd(){  cd $(ciexyz-dir); }

ciexyz-pdf(){
   local path=$(ciexyz-pdfpath)
   mkdir -p $(dirname $path)
   [ ! -f "$path" ] && curl -L http://www.ppsloan.org/publications/XYZJCGT.pdf -o $path
   open $path
}

ciexyz-pdfpath(){
   echo $LOCAL_BASE/env/graphics/ciexyz/XYZJCGT.pdf
}
ciexyz-lib(){
   echo $LOCAL_BASE/env/graphics/ciexyz/ciexyz.dylib
}

ciexyz-make-ctypes()
{
   ciexyz-cd
   local lib=$(ciexyz-lib)
   mkdir -p $(dirname $lib) 
   clang ciexyz.c -shared -o $lib
}

ciexyz-make-ufunc()
{
   type $FUNCNAME
   ciexyz-cd
   sudo python setup.py install
   sudo rm -rf build


}

ciexyz-make()
{
   ciexyz-make-ufunc
   ciexyz-test-ufunc
}

ciexyz-test-ufunc()
{
   ciexyz-cd
   python ciexyz_ufunc.py 
}


