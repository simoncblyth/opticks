cfh-rel(){      echo ana ; }
cfh-src(){      echo ana/cfh.bash ; }
cfh-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cfh-src)} ; }
cfh-vi(){       vi $(cfh-source) ; }
cfh-usage(){ cat << \EOU

Random access to plot pages of qwn/irec AB comparison histos::

   cfh-;cfh "TO SC BT BT BT BT [SA]"
   cfh-;cfh "TO BT BT BT BT DR BT BT BT BT BT BT BT BT [SA]"
   cfh-;cfh "TO RE [BT] BT BT BT SA"

Plotting histograms selected by chi2 and stats::

   cfh-;cfh-chi2sel    

To change binning adjust evt.py binscale and run::

    tconcentric-;tconcentric-rehist

Old way::

   cfh concentric/1/TO_BT_BT_BT_BT_SA/0/X
   cfh /tmp/blyth/opticks/CFH/concentric/1/TO_BT_BT_BT_BT_SA/0/X

   cfh-;cfh concentric/1/TO_BT_BT_BT_BT_SA/0/XYZT

   cfh-;cfh concentric/1/TO_SC_BT_BT_BT_BT_SA/6/XYZT
   cfh-;cfh --rehist concentric/1/TO_SC_BT_BT_BT_BT_SA/6/XYZT


EOU
}

cfh(){ ipython -i $(which cfh.py) -- "$*" ; }

cfh-chi2sel()
{
    #ipython -i $(which cfh.py) -- --chi2sel --chi2selcut 1.2 --statcut 1000
    ipython -i $(which cfh.py) -- --chi2sel --chi2selcut 0.7 --statcut 5000
}
cfh-rehist()
{
    ipython -i $(which cfh.py) -- --rehist "$*"
}

cfh-env(){
    olocal-
    opticks-
}

