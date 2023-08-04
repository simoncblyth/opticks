storch_FillGenstep_usage(){ cat << EOU
storch_FillGenstep.sh
=======================

This is sourced from scripts setting up environment
for torch photon generation 

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
    check=rain_line
    #check=up_rain_line
    #check=escape
    #check=rain_dynode
    #check=rain_dynode_diag
    #check=lhs_window_line
    #check=lhs_reflector_line
    #check=lhs_reflector_point
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

    elif [ "$CHECK" == "rain_dynode_diag" ]; then

        ttype=line
        radius=120   
        pos=0,0,-50
        mom=1,0,-1

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

