/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// https://gist.github.com/jpthompson23/d63415852e2ead56fe74

#include <iostream>
#include <cmath>
#include <functional>

struct NewtonSolver 
{
    std::function<float(const float)> f;
    std::function<float(const float)> fprime;
    const float tol;
    int maxstep ; 

    NewtonSolver(
        float(& _f)(const float), 
        float(& _fprime)(const float), 
        const float _tol,
        int _maxstep
        )
        :
        f(_f), 
        fprime(_fprime), 
        tol(_tol),
        maxstep(_maxstep)
     {
     }

    bool not_there_yet(float& x) const
    {
        float old_x = x;
        x -= f(x)/fprime(x);
        return std::abs(x - old_x) > std::abs(x)*tol ;
    } 

    void operator()(float& x) const 
    {
        int step = 0 ; 
        while(not_there_yet(x) && step < maxstep ) step++ ;
    }
};


float f(const float x) 
{
    return x*x + 2*x - 3;
}

float fprime(const float x) 
{
    return 2*x + 2;
}



int main() 
{
    const float tol = 0.001;
    float x = -1000;
    NewtonSolver newton(f, fprime, tol, 10);
    newton(x);
    std::cout << "Final answer: " << x << std::endl;
    return 0;
}
