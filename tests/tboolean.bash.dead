
tboolean-dead(){ cat <<EON

CsgInBox declared dead ? Usurped by py config 

* but not quite ready to delete the code...


EON
}



tboolean-csg-shells3-notes(){ cat << EON

* https://en.wikipedia.org/wiki/Tetrahedron#Formulas_for_a_regular_tetrahedron

Tetrahedron: (1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)

* four difference of sphere bubbles centered on tetrahedron vertices

Causes errors at 3-way intersections


                  U

        U                    U

   D         D          D         D
 s   s     s   s      s   s     s   s


Hmm quite difficult to construct height 3 trees using levelorder 


* this motivated development of python based CSG config 

EON
}


tboolean-csg-shells3()
{
    local boundary=$(tboolean-object)

    local o=200
    local i=190
    local s=100   # tetrahedron vertex distance, at 150 the shells dont overlap 
    local t=100

    local shape=sphere

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=union        parameters=0,0,0,500           boundary=$boundary

                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary

                      node=difference   parameters=0,0,0,500           boundary=$boundary
                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=difference   parameters=0,0,0,500           boundary=$boundary
                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=$shape       parameters=$s,$s,$s,$o         boundary=$boundary
                      node=$shape       parameters=$s,$s,$s,$i         boundary=$boundary

                      node=$shape       parameters=$s,-$s,-$s,$o       boundary=$boundary
                      node=$shape       parameters=$s,-$s,-$s,$i       boundary=$boundary

                      node=$shape       parameters=-$t,$t,-$t,$o       boundary=$boundary
                      node=$shape       parameters=-$t,$t,-$t,$i       boundary=$boundary

                      node=$shape       parameters=-$t,-$t,$t,$o       boundary=$boundary
                      node=$shape       parameters=-$t,-$t,$t,$i       boundary=$boundary
 
                      )

    echo "$(join _ ${test_config[@]})" 
}




tboolean-csg-shells3-alt()
{
    local boundary=$(tboolean-object)

    local o=200
    local i=190
    local s=100 

    local shape=sphere

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary


                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary


                      node=$shape       parameters=$s,$s,$s,$o         boundary=$boundary
                      node=$shape       parameters=$s,-$s,-$s,$o       boundary=$boundary

                      node=$shape       parameters=-$s,$s,-$s,$o       boundary=$boundary
                      node=$shape       parameters=-$s,-$s,$s,$o       boundary=$boundary
                     
 
                      node=$shape       parameters=$s,-$s,-$s,$i       boundary=$boundary
                      node=$shape       parameters=$s,$s,$s,$i         boundary=$boundary

                      node=$shape       parameters=-$s,$s,-$s,$i       boundary=$boundary
                      node=$shape       parameters=-$s,-$s,$s,$i       boundary=$boundary
 
                      )

    echo "$(join _ ${test_config[@]})" 
}




tboolean-csg-shells2-notes(){ cat << EON

* difference of spheres makes a shell
* intersection of shells makes a ring that has a twisty surface
* difference of overlapping shells makes a shell with a ringlike cut 
* union makes a compound shell with crossed surfaces 
* difference of unions of offset spheres makes a kidney bean shell 

* so far have not seen errors from height 2 trees like these


           D
       U       U
     s   s   s   s
    

EON
}



tboolean-csg-shells2()
{
    local boundary=$(tboolean-object)

    local o=200
    local i=190
    local s=50

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=difference   parameters=0,0,0,500           boundary=$boundary

                      node=union        parameters=0,0,0,500           boundary=$boundary
                      node=union        parameters=0,0,0,500           boundary=$boundary

                      node=sphere       parameters=-$s,-$s,-$s,$o      boundary=$boundary
                      node=sphere       parameters=$s,$s,$s,$o         boundary=$boundary

                      node=sphere       parameters=-$s,-$s,-$s,$i      boundary=$boundary
                      node=sphere       parameters=$s,$s,$s,$i         boundary=$boundary
 
                      )

    echo "$(join _ ${test_config[@]})" 
}




tboolean-csg-four-box-minus-sphere()
{
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local radius=200
    local s=100
    #local s=200  # no error when avoid overlap between subtrees 

    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=box          parameters=$s,$s,-$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=$s,$s,-$s,200           boundary=$(tboolean-object)

                      node=box          parameters=-$s,-$s,-$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=-$s,-$s,-$s,200           boundary=$(tboolean-object)

                      node=box          parameters=$s,-$s,$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=$s,-$s,$s,200           boundary=$(tboolean-object)

                      node=box          parameters=-$s,$s,$s,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=-$s,$s,$s,200           boundary=$(tboolean-object)
 
                      )

    echo "$(join _ ${test_config[@]})" 
}


tboolean-csg-triplet()
{
    local material=$(tboolean-material)
    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=intersection   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=box          parameters=0,0,0,150           boundary=$(tboolean-object)
                      node=sphere       parameters=0,0,0,200           boundary=$(tboolean-object)
 
                      )

    echo "$(join _ ${test_config[@]})" 
}


tboolean-csg-two-box-minus-sphere-interlocked()
{
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local test_config=(
                      mode=CsgInBox
                      name=$FUNCNAME
                      analytic=1
                      offsets=0,1
                      node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)

                      node=union        parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)
                      node=difference   parameters=0,0,0,500           boundary=$(tboolean-object)

                      node=box          parameters=100,100,-100,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=100,100,-100,200           boundary=$(tboolean-object)

                      node=box          parameters=0,0,100,$inscribe     boundary=$(tboolean-object)
                      node=sphere       parameters=0,0,100,200           boundary=$(tboolean-object)
 
                      )
    echo "$(join _ ${test_config[@]})" 
}



tboolean-testconfig()
{
    #tboolean-csg-four-box-minus-sphere
    #tboolean-csg-shells2
    #tboolean-csg-shells3
    #tboolean-csg-shells3-alt
    #tboolean-csg
    tboolean-csg-triplet


}

