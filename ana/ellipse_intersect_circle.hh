#pragma once

/**
ellipse_intersect_circle.hh
-----------------------------

Numerical intersection of an ellipse and a circle 
returning 0,1,2,3 or 4 intersection points.

Developed within opticks/ana, see::

    ana/ellipse_intersect_circle.hh
    ana/ellipse_intersect_circle.cc
    ana/shape.py
    analytic/gdml.py
    ana/gplt.py
    ana/gpmt.py

Started from::

    def points_inside_circle(points, center, radius):
        """ 
        :param points: (n,2) array of points
        :param center: (2,) coordinates of circle center
        :param radius:
        :return mask: boolean array of dimension (n,2) indicating if points are within the circle 
        """
        return np.sqrt(np.sum(np.square(points-center),1)) - radius < 0.  

**/

#include <stdio.h>
#include <math.h>
#include <assert.h>

//#define WITH_MASK 1

struct Point
{
    double x ; 
    double y ; 
    int    i ; 
    int    n ; 

    void dump(const char* msg) const
    {
        printf("Point %s : i %d n %d  (%10.4f, %10.4f)  \n", msg, i, n, x, y );  
    }

    void copy(const Point& other)
    {
        x = other.x ; 
        y = other.y ; 
        i = other.i ; 
        n = other.n ; 
    }

    void set(double x_, double y_, int i_, int n_ )
    {
        x = x_ ; 
        y = y_ ;
        i = i_ ; 
        n = n_ ;  
    }

    static Point make(double px, double py, int i=-1, int n=-1)
    {
        Point p ; 
        p.x = px; 
        p.y = py;
        p.i = i ; 
        p.n = n ; 
        return p ; 
    }

    double distance(const Point& other) const 
    {
        double dx = other.x - x ; 
        double dy = other.y - y ; 
        return sqrt(dx*dx + dy*dy) ; 
    }
};

struct Circle
{
    Point center ; 
    double radius ; 

    static Circle make(double cx, double cy, double r )
    {
        Circle c ; 
        c.center = Point::make(cx, cy); 
        c.radius = r ; 
        return c ; 
    }

    bool is_point_inside( const Point& point)
    { 
        double d = center.distance(point) ; 
        bool inside = d - radius < 0. ; 
        return inside ; 
    }
};


struct Mask
{
    bool* a ;
    int n ; 

    Mask(int n_)
       :
       a(new bool[n_]),
       n(n_)
    {
    } 


    int first() const 
    {
        for(int i=0 ; i < n ; i++) if(a[i]) return i ; 
        return -1 ; 
    }

    int last() const 
    {
        for(int i=0 ; i < n ; i++) if(a[n-1-i]) return n-1-i ; 
        return -1 ; 
    }
    
    int count() const 
    {
        int tot = 0 ; 
        for(int i=0 ; i < n ; i++) if(a[i]) tot += 1 ; 
        return tot ; 
    } 

    void dump(const char* msg) const
    {
        printf("Mask %s n %d first %d last %d count %d \n", msg, n, first(), last(), count() );  
    }


};


struct Ellipse
{
    Point center ; 
    double ax ; 
    double ay ; 

    static Ellipse make( double px, double py, double ax_, double ay_ )
    {
         Ellipse e ; 
         e.center = Point::make( px, py); 
         e.ax = ax_ ; 
         e.ay = ay_ ;
         return e ;  
    } 

    void get_point( Point& p, int i, int n ) const 
    {
         /**
         With typical ranges, do not get to fraction 1 : so does not reach 2pi and the initial point

            range of i        :  0 -> n-1 
            range of fraction :  0/n -> (n-1)/n  : 0 -> 1-1/n        

         **/
         assert( i > -1 && i < n ); 

         double fraction = double(i)/double(n) ; 
         double t = M_PI*2.*fraction ; 

         p.x = ax*cos(t) + center.x ; 
         p.y = ay*sin(t) + center.y ; 
         p.i = i ; 
         p.n = n ; 
    }

    void dump_point( const char* msg, int i, int n ) const 
    {
        Point point ; 
        get_point(point, i, n ); 
        printf("%s  i %3d/%d   point %10.4f %10.4f \n", msg, i, n, point.x, point.y ); 
    } 

    void dump_points( const Mask& mask ) const 
    {
         for(int i=0 ; i < mask.n ; i++)
         {
             if(!mask.a[i]) continue ; 
             dump_point("", i, mask.n ); 
         } 
    } 

};


struct Intersect
{
    Point* p ; 
    int mx ;
    int count ;  

    static Intersect make(int mx_)
    {
         Intersect isect ; 
         isect.p = new Point[mx_] ; 
         isect.mx = mx_ ; 
         isect.count = 0 ;  
         return isect ; 
    }

    void add(const Point& point)
    {
        assert( count < mx );  
        p[count].copy(point) ;
        count += 1 ;  
    } 

    void dump(const char* msg)
    {
        printf("Intersect %s : count %d crossings  \n", msg, count ); 
        for(int i=0 ; i < count ; i++)
        {
            p[i].dump("p"); 
        } 

    }

}; 



struct Ellipse_Intersect_Circle
{
    Ellipse ellipse ; 
    Circle circle ;
    int n ;  
#ifdef WITH_MASK
    Mask* mask ; 
#endif
    Intersect intersect ; 
    bool verbose ; 

    enum {
       INSIDE, 
       OUTSIDE
    };


    static Ellipse_Intersect_Circle make(double e_cx, double e_cy, double e_ax, double e_ay,  double c_cx, double c_cy, double c_r, int n, bool verbose )
    {
        Ellipse_Intersect_Circle ec ; 
        ec.ellipse = Ellipse::make( e_cx, e_cy, e_ax, e_ay ); 
        ec.circle = Circle::make( c_cx, c_cy, c_r ); 
        ec.n = n ; 
        ec.intersect = Intersect::make(4) ; 
        ec.verbose = verbose ; 

        int ni = ec.find_intersects(); 
        if(verbose)
        {
            printf("Ellipse_Intersect_Circle found %d intersects\n", ni ); 
            ec.intersect.dump("EC") ; 
        }


#ifdef WITH_MASK
        ec.mask = new Mask(n) ; 
        ec.find_mask(); 
#endif

        return ec ; 
    }


    int find_intersects()
    {
         /*
         A direct comparison of states between the first 
         and last is done in order to "close" the ellipse.

         Crossings are checked for all points around the ellipse
         traversed in an anti-clockwise direction starting at right hand extremity
         and are collected in that order.

         */

         Point e ;
         ellipse.get_point(e, 0, n ); 
         int s0 = circle.is_point_inside(e) ? INSIDE : OUTSIDE ;  
         int s_prev = s0 ; 

         for(int i=1 ; i < n ; i++)
         {
             ellipse.get_point(e, i, n ); 
             int s = circle.is_point_inside(e) ? INSIDE : OUTSIDE ;  
             if( s != s_prev ) intersect.add(e) ; 
             s_prev = s ; 
         } 
       
         if( s_prev != s0 ) intersect.add(e) ;  

         return intersect.count ; 
    }

#ifdef WITH_MASK
    int find_mask() 
    {
         Point e ;
         int num_inside = 0 ; 
         for(int i=0 ; i < mask->n ; i++)
         {
             ellipse.get_point(e, i, mask->n ); 
             bool inside = circle.is_point_inside(e); 
             mask->a[i] = inside ; 
             if(inside) num_inside += 1 ; 
         } 
         if(verbose) mask->dump("ellipse points inside circle");  
         return num_inside ; 
    }
#endif

}; 


