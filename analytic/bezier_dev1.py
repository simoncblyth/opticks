#!/usr/bin/env python
"""

Cubic Bezier polynomial interpolating from P0 to P3 as u 
goes from 0 to 1 controlled by P1, P2::

    B(u) = P0*(1-u)**3 + P1*3*u*(1-u)**2 + P2*3*u**2*(1-u) + P3*u**3   

To apply to surface of revolution (rr,z) in range z1 to z2, equate 

         (z - z1)
    u =  --------         u = 0 at z=z1  u = 1 at z=z2
          (z2 - z1)

Or more in spirit of Bezier decide on begin/end points and 
control points

::

    (z1, rr1) 
    (cz1, crr1)
    (cz2, crr2)
    (z2, rr2) 


* https://stackoverflow.com/questions/246525/how-can-i-draw-a-bezier-curve-using-pythons-pil


::

    In [6]: bezier([0,1])
    Out[6]: [(50, 100), (100, 50)]

    In [7]: bezier([0,0.5,1])
    Out[7]: [(50, 100), (77.5, 77.5), (100, 50)]

    In [8]: bezier([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    Out[8]: 
    [(50, 100),
     (55.900000000000006, 95.9),
     (61.60000000000001, 91.60000000000002),
     (67.1, 87.1),
     (72.4, 82.4),
     (77.5, 77.5),
     (82.4, 72.4),
     (87.1, 67.1),
     (91.60000000000001, 61.6),
     (95.89999999999999, 55.9),
     (100, 50)]


"""



def pascal_row(n):
    # This returns the nth row of Pascal's Triangle
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result)) 
    return result


def make_bezier(xys):
    """
    :param xys: sequence of 2-tuples (Bezier control points)
    :return func: call it over t parameter iterable
    
    Uses the generalized formula for bezier curves
    http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization

    For cubic bezier with 4 points combinations is just (1,3,3,1)
    """
    n = len(xys)
    combinations = pascal_row(n-1)   

    def bezier(ts):
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    pass
    return bezier




from PIL import Image
from PIL import ImageDraw


def bezier_heart():
    im = Image.new('RGBA', (100, 100), (0, 0, 0, 0)) 
    draw = ImageDraw.Draw(im)
    ts = [t/100.0 for t in range(101)]

    xys = [(50, 100), (80, 80), (100, 50)]
    bezier = make_bezier(xys)
    points = bezier(ts)

    xys = [(100, 50), (100, 0), (50, 0), (50, 35)]
    bezier = make_bezier(xys)
    points.extend(bezier(ts))

    xys = [(50, 35), (50, 0), (0, 0), (0, 50)]
    bezier = make_bezier(xys)
    points.extend(bezier(ts))

    xys = [(0, 50), (20, 80), (50, 100)]
    bezier = make_bezier(xys)
    points.extend(bezier(ts))

    draw.polygon(points, fill = 'red')
    im.save('out.png')



if __name__ == '__main__':

    xys = [(50,100), (80,80), (100,50) ]
    bezier =  make_bezier(xys)


