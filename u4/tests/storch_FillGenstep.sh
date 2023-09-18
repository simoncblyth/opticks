storch_FillGenstep_usage(){ cat << EOU
storch_FillGenstep.sh
=======================

This is sourced from scripts to setup envvars
that configure torch photon generation 

Some of this config such as CHECK 
may be overridden by callers, for example:

* ~/opticks/g4cx/tests/G4CXTest.sh 


Control vars
-------------

* LAYOUT : one_pmt/two_pmt 
* CHECK  : many possibilites different within each LAYOUT

Output var exports
--------------------

* CHECK 
* storch_FillGenstep_type
* storch_FillGenstep_radius
* storch_FillGenstep_pos
* storch_FillGenstep_mom


EOU
}

layout=one_pmt
export LAYOUT=${LAYOUT:-$layout}

if [ "$LAYOUT" == "one_pmt" ]; then 

    #check=rain_disc
    #check=rain_line
    #check=rain_line_205
    #check=rain_point_xpositive_0
    #check=tub3_side_line
    #check=rain_point_xpositive_100
    #check=up_rain_line
    #check=escape
    #check=rain_dynode
    #check=rain_dynode_diag
    #check=lhs_window_line
    #check=lhs_reflector_line
    #check=lhs_reflector_point
    #check=rectangle_inwards
    check=mask_tail_diagonal_line

    export CHECK=${CHECK:-$check}  # CAUTION: this is duplicated for other layouts

    if [ "$CHECK" == "rain_disc" ]; then

        ttype=disc 
        pos=0,0,195
        mom=0,0,-1
        radius=250
        # approx PMT extents : xy -255:255, z -190:190

    elif [ "$CHECK" == "rain_line" ]; then

        ttype=line
        pos=0,0,195    ## 190 grazes HAMA apex
        radius=260     # standand for line from above,  280 hangsover  
        mom=0,0,-1   

    elif [ "$CHECK" == "rain_line_205" ]; then

        ttype=line
        pos=0,0,205    ## increase for shooting nmskLogicMaskVirtual, hmskLogicMaskVirtual
        radius=260     # standand for line from above,  280 hangsover  
        mom=0,0,-1   

    elif [ "${CHECK:0:21}" == "rain_point_xpositive_" ]; then

        xpos=${CHECK:21}
        ttype=point
        pos=$xpos,0,195    ## 190 grazes HAMA apex
        mom=0,0,-1   
        radius=0  

    elif [ "${CHECK:0:21}" == "rain_point_xnegative_" ]; then

        xpos=${CHECK:21}
        ttype=point
        pos=-$xpos,0,195    ## 190 grazes HAMA apex
        mom=0,0,-1   
        radius=0  

    elif [ "${CHECK:0:16}" == "circle_outwards_" ]; then

        ttype=circle
        radius=${CHECK:16}   # +ve radiys for outwards
        pos=0,0,0

    elif [ "${CHECK:0:15}" == "circle_inwards_" ]; then

        ttype=circle
        radius=-${CHECK:15}   # -ve radius for inwards
        pos=0,0,0

    elif [ "$CHECK" == "rectangle_inwards" ]; then

        ttype=rectangle
        pos=0,0,0 
        zenith=-205,205
        azimuth=-290,290

    elif [ "$CHECK" == "up_rain_line" ]; then

        ttype=line
        radius=260
        pos=0,0,-195  
        mom=0,0,1        

    elif [ "$CHECK" == "escape" ]; then

        ttype=point
        pos=0,0,100 
        mom=0,0,1
        radius=0

    elif [ "$CHECK" == "rain_dynode" ]; then

        ttype=line
        radius=120    # focus on HAMA dynode
        pos=0,0,-50
        mom=0,0,-1


    elif [ "$CHECK" == "rain_down_100" ]; then

        ttype=line
        radius=100    
        pos=0,0,-100
        mom=0,0,-1

    elif [ "$CHECK" == "rain_dynode_diag" ]; then

        ttype=line
        radius=120   
        pos=0,0,-50
        mom=1,0,-1

    elif [ "$CHECK" == "tub3_side_line" ]; then

        ttype=line
        radius=60     
        pos=-60,0,-20   ## line from (-60,0,-80) to (-60,0,40) 
        mom=1,0,0

    elif [ "$CHECK" == "lhs_window_line" ]; then

        ttype=line
        radius=95     
        pos=-300,0,95   ## line from (-300,0,0) to (-300,0,190) 
        mom=1,0,0

    elif [ "$CHECK" == "lhs_reflector_line" ]; then

        ttype=line
        radius=95
        pos=-300,0,-95   ## line from (-300,0,0) to (-300,0,-190)
        mom=1,0,0        

    elif [ "$CHECK" == "lhs_reflector_point" ]; then

        ttype=point
        pos=-300,0,-10     ## PMT left below cathode at Z=0, for shooting the reflector 
        mom=1,0,0
        radius=0

    elif [ "$CHECK" == "mask_tail_diagonal_line" ]; then

        intent="point symmetrically placed to tareget outside of nmskTail"
        ttype=line
        radius=50   
        pos=-214,0,-127
        mom=1,0,1

    else
         echo $BASH_SOURCE : ERROR LAYOUT $LAYOUT CHECK $CHECK IS NOT HANDLED
    fi 


elif [ "$LAYOUT" == "two_pmt" ]; then

    check=right_line
    export CHECK=${CHECK:-$check}  # CAUTION: this is duplicated for other layouts

    if [ "$CHECK" == "right_line" ]; then 
        ttype=line
        radius=250
        pos=0,0,0   
        mom=1,0,0        
    else
        echo $BASH_SOURCE : ERROR LAYOUT $LAYOUT CHECK $CHECK IS NOT HANDLED
    fi
fi 

export storch_FillGenstep_type=$ttype
export storch_FillGenstep_radius=$radius
export storch_FillGenstep_pos=$pos
export storch_FillGenstep_mom=$mom
[ -n "$zenith" ]  && export storch_FillGenstep_zenith=$zenith
[ -n "$azimuth" ] && export storch_FillGenstep_azimuth=$azimuth


