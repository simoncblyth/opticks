cfh-rel(){      echo ana ; }
cfh-src(){      echo ana/cfh.bash ; }
cfh-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cfh-src)} ; }
cfh-vi(){       vi $(cfh-source) ; }
cfh-usage(){ cat << \EOU

Random access to qwn/irec AB comparison histograms and chi2 within AB single line selections::

   cfh concentric/1/TO_BT_BT_BT_BT_SA/0/X
   cfh /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X

   cfh-;cfh concentric/1/TO_BT_BT_BT_BT_SA/0/XYZT

   cfh-;cfh concentric/1/TO_SC_BT_BT_BT_BT_SA/6/XYZT
   cfh-;cfh --rehist concentric/1/TO_SC_BT_BT_BT_BT_SA/6/XYZT


   cfh-;cfh --chi2sel    # plotting pages selected by distrib chi2 sum greater than cut 


EOU
}

cfh()
{
    ipython -i $(which cfh.py) -- $*
}

cfh-env(){
    olocal-
    opticks-
}

