#include "Animator.hh"
#include "NPY.hpp"


int main()
{
     unsigned int N = 1000 ; 
     NPY<float>* npy = NPY<float>::make_vec4(N, 1, 0.f); 

     Animator anim(200);
     for(unsigned int i=0 ; i < N ; i++)
     {
         bool bump(false);
         float frac = anim.step(bump);
         printf("%5d bump? %d %16s %10.4f \n", i, bump, anim.description(), frac)   ;
         if( i == 100 ) anim.scalePeriod(0.5);
         if( i == 400 ) anim.scalePeriod(2.0);

         npy->setQuad(i, 0, float(i), float(frac), 0.f, 0.f );
     }

     npy->save("/tmp/animator.npy");
     return 0 ;
}

/*
Aim is for a nice sawtooth with no glitches::

    In [1]: a = np.load("/tmp/animator.npy")

    In [2]: plt.ion()

    In [3]: plt.plot(a[:,0,0], a[:,0,1])
    Out[3]: [<matplotlib.lines.Line2D at 0x111587610>]


*/
