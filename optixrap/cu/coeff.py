#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#


from sympy import expand, symbols, simplify, collect, factor, solve, Pow


def get_coeff_(expr, var, n):
    c = range(n)
    for i in range(n-1,-1,-1):
        c[i] = collect(expand(expr), var ).coeff(var,i)
    pass
    return c


def get_coeff(expr, var, n):
    c = range(n)
    for i in range(n-1,-1,-1):
        c[i] = factor(collect(expand(expr), var ).coeff(var,i))
    pass
    return c

def print_coeff(c, msg="coeff"):
    print msg
    for i in range(len(c)):
        print "c[%d]:%r " % (i, c[i])
    pass

def subs_coeff(c, sub):
    for i in range(len(c)):
        c[i] = c[i].subs(sub)
    pass


def expr_coeff(c, var):
    ex = None
    for i in range(len(c)):
        if ex is None:
            ex = Pow(var, i)*c[i]
        else:
            ex += Pow(var, i)*c[i]
        pass
    pass
    return ex


