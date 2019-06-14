ezgdml-source(){ echo $BASH_SOURCE ; }
ezgdml-vi(){ vi $(ezgdml-source)  ; }
ezgdml-env(){  olocal- ; opticks- ; geocache- ;  }
ezgdml-usage(){ cat << EOU

EOU
}

ezgdml-path(){ echo g4codegen/tests/x016.gdml ; }

ezgdml--()
{
   geocache-kcd
   ipython -i $(which ezgdml.py) -- $(ezgdml-path)
}



