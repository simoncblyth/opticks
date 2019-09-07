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

"""

* http://blog.moertel.com/posts/2013-05-11-recursive-to-iterative.html


"""

ff = []

def factorial_0(n):
    if n < 2:
        return 1
    return n * factorial_0(n - 1)
pass
ff.append(factorial_0)

def factorial_1(n, acc=1):
    """
    Tail call, last line just does the recursive call 
    """
    if n < 2:
        return 1 * acc
    return factorial_1(n - 1, acc * n)
pass
ff.append(factorial_1)


def factorial_2(n, acc=1):
    """
    Stuff in loop
    """ 
    while True:
        if n < 2:
            return 1 * acc
        return factorial_2(n - 1, acc * n)
        break
ff.append(factorial_2)


def factorial_3(n, acc=1):
    """
    Replace all recursive tail calls f(x=x1, y=y1, ...) with (x, y, ...) = (x1, y1, ...); continue
    """
    while True:
        if n < 2:
            return 1 * acc
        (n, acc) = (n - 1, acc * n)
        continue
        break
ff.append(factorial_3)


def factorial_5(n, acc=1):
    while n > 1:
        (n, acc) = (n - 1, acc * n)
    return acc
ff.append(factorial_5)




if __name__ == '__main__':

    for f in ff:
        v = f(5)
        print f.__name__, v
        assert v == 120


