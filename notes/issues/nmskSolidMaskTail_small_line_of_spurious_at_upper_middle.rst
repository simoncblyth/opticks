nmskSolidMaskTail_small_line_of_spurious_at_upper_middle
============================================================


::

     geom_ # set default to nmskSolidMaskTail

     gc
     GEOM=nmskSolidMaskTail ./translate.sh 


::

     gx
     ./gxt.sh         # workstation
     ./gxt.sh grab    # laptop
     ./gxt.sh ana     # laptop


Central spurious are in line x=-1:1 z=-39 

Morton finder fails to select them as too many together, so increase the SPURIOUS_CUT to find them::

    MASK=t SPURIOUS=1 ./gxt.sh ana 
    MASK=t SPURIOUS=4 ./gxt.sh ana 

Looking closer at the sides edge shows nasty lips, with disconnected lines of spurious at z=-39.

Will need to look at constituents, add the below to gc:mtranslate.sh::

    geomlist_nmskSolidMaskTail(){ cat << EOL
    nmskSolidMaskTail

    nmskTailOuter
    nmskTailOuterIEllipsoid
    nmskTailOuterITube
    nmskTailOuterI
    nmskTailOuterIITube

    nmskTailInner
    nmskTailInnerIEllipsoid
    nmskTailInnerITube
    nmskTailInnerI
    nmskTailInnerIITube 

    EOL
    }





