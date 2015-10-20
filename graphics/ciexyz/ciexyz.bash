# === func-gen- : graphics/ciexyz/ciexyz fgp graphics/ciexyz/ciexyz.bash fgn ciexyz fgh graphics/ciexyz
ciexyz-src(){      echo graphics/ciexyz/ciexyz.bash ; }
ciexyz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ciexyz-src)} ; }
ciexyz-vi(){       vi $(ciexyz-source) ; }
ciexyz-env(){      elocal- ; }
ciexyz-usage(){ cat << EOU


Simple Analytic Approximations to the CIE XYZ Color Matching Functions

* http://www.ppsloan.org/publications/XYZJCGT.pdf

Related

* https://www.fourmilab.ch/documents/specrend/



EOU
}
ciexyz-dir(){ echo $(env-home)/graphics/ciexyz ; }
ciexyz-cd(){  cd $(ciexyz-dir); }
