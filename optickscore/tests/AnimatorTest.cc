#include "Animator.hh"
#include "NPY.hpp"


int main()
{
     unsigned int N = 1000 ; 
     NPY<float>* npy = NPY<float>::make(N, 1, 4);
     npy->fill(0.f); 

     float target(-1.f) ;
     Animator anim(&target, 200);
     anim.setMode(Animator::NORM);

     for(unsigned int i=0 ; i < N ; i++)
     {
         bool bump(false);
         anim.step(bump);
         printf("%5d bump? %d %16s %10.4f \n", i, bump, anim.description(), target)   ;
         if( i == 100 ) anim.setMode(Animator::FAST);
         if( i == 400 ) anim.setMode(Animator::SLOW8);

         npy->setQuad(i, 0, float(i), float(target), 0.f, 0.f );
     }

     npy->save("$TMP/animator.npy");
     return 0 ;
}

/*
Aim is for a nice sawtooth with no glitches::

    In [1]: a = np.load("$TMP/animator.npy")

    In [2]: plt.ion()

    In [3]: plt.plot(a[:,0,0], a[:,0,1])
    Out[3]: [<matplotlib.lines.Line2D at 0x111587610>]


*/
