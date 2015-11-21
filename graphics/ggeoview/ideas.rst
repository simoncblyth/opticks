Ideas
=======


Geometry
----------

Prism TIR
~~~~~~~~~~~

Implement Prism (triangular half cube) 


Cartesian Oval Surface of Revolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://en.wikipedia.org/wiki/Cartesian_oval (Hecht p130)

::

  l0 n1 + li n2 = constant


Practicalities
~~~~~~~~~~~~~~~~

* enable test geometry to somehow fallback to triangulated when no anlytic




Optical Tests
--------------


Spherical Interface Refraction
------------------------------------


p133 Hecht, refraction at spherical interface::

    .                                  
                                 / 
                                A    
                               /
          SA = l0             /           AP = li
                             /
                             |
    S -----------------------V             C        P 
    .                        .                      .
    |         so             |          si          |     
        


    S : point source
    C : center of sphere
    P : point where ray crosses axis

    phi        : angle SCA
    180. - phi : angle PCA 


    law of cosines for triangles SAC and ACP
    and angle relation: SCA + PCA = pi
     
    optical path length,  OPL = n1 l0 + n2 li

    Fermats principle, dOPL/dx = 0   (derivative with position variable) 

    n1 R (s0 + R) sin(phi)      n2 R (si - R ) sin(phi) 
    ----------------------  -   ------------------------  = 0
        2 l0                             2 li

    yielding relation between parameters of ray going from S to P via refraction at spherical interface

    n1     n2       1  /  n2 si     n1 so   \
    --  +  --   =   -- |  ----- -  -------  | 
    l0     li       R  \    li       l0     /

    small angle assumption A close to V,  cos(phi) ~ 1  sin(phi) ~ phi 
    (this assumption corresponds to paraxial rays and is known as Gaussian Optics)

    l0 ~ s0   
    li ~ si

    n1     n2      n2 - n1
    --  +  --  =   -------
    s0     si         R




Thin Gaussian Lens
~~~~~~~~~~~~~~~~~~~


p138 Hecht, spherical lens assuming small angles from optical axis (paraxial rays)


                   /|\
                  / | \ 
                 /  |  \
        C2      V1  |  V2       C1 
                 \  |  /             
                  \ | /
                   \|/
             
        |       |   d   |        |

        |     R2        |

                |        R1      |
                          

     C1 - R1 + d = C2 + R2 

              d  = R2 + R1 - (C2 - C1)  




    nm      nm                /  1     1   \         nl d
    ---  +  ---  =  (nl - nm) |  -  -  -   |   +  ------------
    so1     si2               \  R1    R2  /      (si1 - d)sil


Thin lens assumption removes the d term, and simplify with air/vacuum nm=1 get
relation between object and image distances::

     1      1         1                /  1      1   \
     --  +  ---   =   --  =   (nl - 1) |  --  -  --  |  
     so     si        f                \  R1     R2  /

                          
                          =   2 (nl - 1 )        for R1 = -R2 = R       
                              -----------
                                   R          

With parallel rays, 1/so = 0::

     si = f =   R / 2(nl - 1)  

For example Vacuum/Pyrex::

    ggv --mat Pyrex   # index 1.458 

    si = f = R * 1.0917

    In [2]: 1./(2*(1.458-1.))
    Out[2]: 1.091703056768559

    In [3]: 700./1.091703056768559
    Out[3]: 641.1999999999999     
        

Pick radius to make focus at edge of box::

    local test_config=(
                 mode=BoxInBox
                 analytic=1

                 shape=B,L

                 boundary=Rock//perfectAbsorbSurface/Vacuum
                 parameters=-1,1,0,700

                 boundary=Vacuum///Pyrex 
                 parameters=641.2,641.2,-600,600

               )    


Visually at least, get the expected focus point.

TODO:

* numerical check of focus coordinates, using the record data, incorporating 
  lens thickness  



Dispersing Prisms
~~~~~~~~~~~~~~~~~~~

Hecht p163, deviation angle as function of prism apex angle, refractive index and incident normal angle.
Minimum deviation occurs where ray traverses symmetrically.

How to define a symmetric prism

* apex angle A, height h, depth d

::

   .                
                    A  (0,h)
                   /|\
                  / | \
                 /  |  \
                /   h   \
               /    |    \ (x,y)   
              M     |     N
             /      |      \
            C-------O-------B   
                           
                  (0,0)     
         (-a/2,0)         (a/2, 0)

   
     angles B = C = (180 - A)/2



                  a/2
     tan(A/2) = --------
                   h

     a/2 = h tan(A/2)

     need plane eqns of faces
                                      
     AB direction : ( 0, h) - (a/2, 0)  = (-a/2, h)    ON direction (h, a/2)
     AC direction : ( 0, h) - (-a/2, 0) = ( a/2, h)    OM direction (h, -a/2)
 
     (-a/2, h ).( h, a/2 ) = 0 



     ON. A = (h, a/2)  . (0, h ) =  ah/2
     OM. A = (h, -a/2) . (0, h ) = -ah/2


     hmm can I calc the planes whilst calulating the bounds... 

     plane N 
         (h, a/2, 0 )     ah/2

     plane M
         (h, -a/2, 0)     -ah/2
    
     plane O
         (0, -1,  0)       0

     plane F
         (0,  0,  1)       d/2

     plane B
         (0,  0, -1)      -d/2





     plane containing 

     A    (0,  h,0)
     B    (a/2,0,0)
     B'   (a/2,0,d)


         


     







Dispersion
~~~~~~~~~~~

* dispersion angle calculation yields refractive index, so
  predict the refractive index as a function of wavelength 
  from the angle and compare 

  * or fabricate a material with a linear refractive index  



