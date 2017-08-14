Cubic Bezier SOR Raytrace Shape Shifting 
============================================


CSG_CUBIC primitive implements a surface of revolution with::

      xx + yy = rr = A zzz + B zz + C z + D

For raw unscaled z, even a small z domain of say -200:200 
(nominally in mm) leads to huge zzz terms. Resulting in 
a need for very small A coeff.


Adopting SolveCubicStrobachPolyFit.h does not fix
----------------------------------------------------

* strobach polyfit method claims to handle very disparate roots, but shape shifting behavior 
  looks the same
 

Shape Shifting from large z
----------------------------------

When viewed closeup looks fine and solid, with disconnected
paraboloid shapes matching mpl expectations from cubic.py 
SOR + profile plotting.

But as back away the sheets merge togther.
Looks like a precision problem when using unscaled z 
to do root finding.


::


    1702 tboolean-cubic(){ TESTCONFIG=$($FUNCNAME- 2>/dev/null)    tboolean-- $* ; } 
    1703 tboolean-cubic-(){  $FUNCNAME- | python $* ; } 
    1704 tboolean-cubic--(){ cat << EOP 
    1705 
    1706 from opticks.ana.base import opticks_main
    1707 from opticks.analytic.csg import CSG  
    1708 
    1709 args = opticks_main(csgpath="$TMP/$FUNCNAME")
    1710 
    1711 CSG.boundary = args.testobject
    1712 CSG.kwa = dict(poly="IM", resolution="50")
    1713 
    1714 container = CSG("box", param=[0,0,0,200], boundary=args.container, poly="MC", nx="20" )
    1715   
    1716 #a = CSG.MakeCubic(A=0.0001, B=2, C=2, D=2, z1=-10, z2=10)   ## 
    1717 
    1718 a = CSG.MakeCubic(A=0.0001, B=2, C=10, D=2, z1=-10, z2=10) 
    1719 
    1720 #zrrs = [[-100,30],[-50,80],[50,30],[100,100]]
    1721 #a = CSG.MakeCubicBezier(zrrs)
    1722 
    1723 
    1724 CSG.Serialize([container, a], args.csgpath )
    1725 
    1726 EOP
    1727 }



Can scaling z avoid the numerical problem ?
-------------------------------------------------

Doing root finding with scaled z (zs) arranged to be within range 0 to 1 ?::

        z  - z1
        -------  =  zs      ( 0 at z=z1, 1 at z=z2)    
        z2 - z1

Hmm BUT the root finding is done in t not z ? 

Procedure is to plug parametric ray eqn in the 
implicit yielding cubic in t.

::

      xx + yy = rr = A zzz + B zz + C z + D


Hmm can also scale x and y using bbox dim... so the 
(xs,ys,zs) at scaled into the unit cube  [-1:1,-1:1,-1:1]


* hmm this is a kinda generic thing all primitives
  have a bbox, could scale x,y,z at the part level 


* :google:`ray trace numerical methods`
* https://www.cs.williams.edu/~morgan/cs371-f14/reading/implicit.pdf






CSG_CUBIC primitive implements a surface of revolution with::

      xx + yy = rr = A zzz + B zz + C z + D




Scaling Polynomial Coeffs for better numerical robustness
------------------------------------------------------------

Actually the problem is general to solving any poly 

* :google:`polynomial scaling coeff numerical`

* https://math.stackexchange.com/questions/1384741/how-to-scale-polynomial-coefficients-for-root-finding-algorithms

* https://math.stackexchange.com/questions/1384331/what-types-of-polynomials-cause-root-finders-to-fail

* http://www.akiti.ca/rpoly_ak1_cpp.html






::

    // rpoly_ak1.cpp - Program for calculating the roots of a polynomial of real coefficients.
    // Written in Microsoft Visual Studio Express 2013 for Windows Desktop
    // 27 May 2014
    //
    // The sub-routines listed below are translations of the FORTRAN routines included in RPOLY.FOR,
    // posted off the NETLIB site as TOMS/493:
    //
    // http://www.netlib.org/toms/493
    //
    // TOMS/493 is based on the Jenkins-Traub algorithm.
    //
    // To distinguish the routines posted below from others, an _ak1 suffix has been appended to them.
    //
    // Following is a list of the major changes made in the course of translating the TOMS/493 routines
    // to the C++ versions posted below:
    // 1) All global variables have been eliminated.
    // 2) The "FAIL" parameter passed into RPOLY.FOR has been eliminated.
    // 3) RPOLY.FOR solves polynomials of degree up to 100, but does not explicitly state this limit.
    //     rpoly_ak1 explicitly states this limit; uses the macro name MAXDEGREE to specify this limit;
    //     and does a check to ensure that the user input variable Degree is not greater than MAXDEGREE
    //     (if it is, an error message is output and rpoly_ak1 terminates). If a user wishes to compute
    //     roots of polynomials of degree greater than MAXDEGREE, using a macro name like MAXDEGREE provides
    //     the simplest way of offering this capability.
    // 4) All "GO TO" statements have been eliminated.
    //
    // A small main program is included also, to provide an example of how to use rpoly_ak1. In this 
    // example, data is input from a file to eliminate the need for a user to type data in via
    // the console.

    #include <iostream>
    #include <fstream>
    #include <cctype>
    #include <cmath>
    #include <cfloat>

    using namespace std;

    #define MAXDEGREE   100
    #define MDP1     MAXDEGREE+1

    void rpoly_ak1(double op[MDP1], int* Degree, double zeror[MAXDEGREE], double zeroi[MAXDEGREE]);
    void Fxshfr_ak1(int L2, int* NZ, double sr, double bnd, double K[MDP1], int N, double p[MDP1], int NN, double qp[MDP1], double* lzi, double* lzr, double* szi, double* szr);
    void QuadSD_ak1(int NN, double u, double v, double p[MDP1], double q[MDP1], double* a, double* b);
    int calcSC_ak1(int N, double a, double b, double* a1, double* a3, double* a7, double* c, double* d, double* e, double* f, double* g, double* h, double K[MDP1], double u, double v, double qk[MDP1]);
    void nextK_ak1(int N, int tFlag, double a, double b, double a1, double* a3, double* a7, double K[MDP1], double qk[MDP1], double qp[MDP1]);
    void newest_ak1(int tFlag, double* uu, double* vv, double a, double a1, double a3, double a7, double b, double c, double d, double f, double g, double h, double u, double v, double K[MDP1], int N, double p[MDP1]);
    void QuadIT_ak1(int N, int* NZ, double uu, double vv, double* szr, double* szi, double* lzr, double* lzi, double qp[MDP1], int NN, double* a, double* b, double p[MDP1], double qk[MDP1], double* a1, double* a3, double* a7, double* d, double* e, double* f, double* g, double* h, double K[MDP1]);
    void RealIT_ak1(int* iFlag, int* NZ, double* sss, int N, double p[MDP1], int NN, double qp[MDP1], double* szr, double* szi, double K[MDP1], double qk[MDP1]);
    void Quad_ak1(double a, double b1, double c, double* sr, double* si, double* lr, double* li);

    void rpoly_ak1(double op[MDP1], int* Degree, double zeror[MAXDEGREE], double zeroi[MAXDEGREE]){

    int i, j, jj, l, N, NM1, NN, NZ, zerok;

    double K[MDP1], p[MDP1], pt[MDP1], qp[MDP1], temp[MDP1];
    double bnd, df, dx, factor, ff, moduli_max, moduli_min, sc, x, xm;
    double aa, bb, cc, lzi, lzr, sr, szi, szr, t, xx, xxx, yy;

    const double RADFAC = 3.14159265358979323846/180; // Degrees-to-radians conversion factor = pi/180
    const double lb2 = log(2.0);    // Dummy variable to avoid re-calculating this value in loop below
    const double lo = FLT_MIN/DBL_EPSILON;
    const double cosr = cos(94.0*RADFAC); // = -0.069756474
    const double sinr = sin(94.0*RADFAC); // = 0.99756405

    if ((*Degree) > MAXDEGREE){
        cout << "\nThe entered Degree is greater than MAXDEGREE. Exiting rpoly. No further action taken.\n";
        *Degree = -1;
        return;
    } // End ((*Degree) > MAXDEGREE)

    //Do a quick check to see if leading coefficient is 0
    if (op[0] != 0){

    N = *Degree;
    xx = sqrt(0.5); // = 0.70710678
    yy = -xx;

    // Remove zeros at the origin, if any
    j = 0;
    while (op[N] == 0){
        zeror[j] = zeroi[j] = 0.0;
        N--;
        j++;
    } // End while (op[N] == 0)

    NN = N + 1;

    // Make a copy of the coefficients
    for (i = 0; i < NN; i++)   p[i] = op[i];

    while (N >= 1){ // Main loop
        // Start the algorithm for one zero
        if (N <= 2){
        // Calculate the final zero or pair of zeros
            if (N < 2){
                zeror[(*Degree) - 1] = -(p[1]/p[0]);
                zeroi[(*Degree) - 1] = 0.0;
            } // End if (N < 2)
            else { // else N == 2
                Quad_ak1(p[0], p[1], p[2], &zeror[(*Degree) - 2], &zeroi[(*Degree) - 2], &zeror[(*Degree) - 1], &zeroi[(*Degree) - 1]);
            } // End else N == 2
            break;
        } // End if (N <= 2)

        // Find the largest and smallest moduli of the coefficients

        moduli_max = 0.0;
        moduli_min = FLT_MAX;

        for (i = 0; i < NN; i++){
            x = fabs(p[i]);
            if (x > moduli_max)   moduli_max = x;
            if ((x != 0) && (x < moduli_min))   moduli_min = x;
        } // End for i

        // Scale if there are large or very small coefficients
        // Computes a scale factor to multiply the coefficients of the polynomial. The scaling
        // is done to avoid overflow and to avoid undetected underflow interfering with the
        // convergence criterion.
        // The factor is a power of the base.

        sc = lo/moduli_min;

        if (((sc <= 1.0) && (moduli_max >= 10)) || ((sc > 1.0) && (FLT_MAX/sc >= moduli_max))){
            sc = ((sc == 0) ? FLT_MIN : sc);
            l = (int)(log(sc)/lb2 + 0.5);
            factor = pow(2.0, l);
            if (factor != 1.0){
                for (i = 0; i < NN; i++)   p[i] *= factor;
            } // End if (factor != 1.0)
        } // End if (((sc <= 1.0) && (moduli_max >= 10)) || ((sc > 1.0) && (FLT_MAX/sc >= moduli_max)))

