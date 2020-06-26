#pragma once

#include <stdio.h>
#include <math.h>
#include <assert.h>

struct Point
{
    double x ; 
    double y ; 
    int    i ; 
    int    n ; 

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
};


struct Mask
{
    bool* a ;
    int n ; 

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
        int first_ = first(); 
        int last_ = last(); 
        int count_ = count(); 

        printf("Mask %s n %d first %d last %d count %d \n", msg, n, first_, last_, count_ );  
    }

    static Mask make(int n )
    {
        Mask m ; 
        m.a = new bool[n] ; 
        m.n = n ; 
        return m ; 
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
         double t = M_PI*2.*double(i)/double(n) ; 
         p.x = ax*cos(t) + center.x ; 
         p.y = ay*sin(t) + center.y ; 
         p.i = i ; 
         p.n = n ; 
    }

    int get_point_first( Point& point, const Mask& mask )
    {
        int first = mask.first(); 
        if(first > -1 ) get_point(point, first, mask.n ); 
        return first ;  
    }

    int get_point_last( Point& point, const Mask& mask )
    {
        int last = mask.last(); 
        if(last > -1 ) get_point(point, last, mask.n ); 
        return last ;  
    }


    int points_inside_circle( Mask& mask, const Circle& circle ) const 
    {
         Point point ; 
         int num_inside = 0 ; 
         for(int i=0 ; i < mask.n ; i++)
         {
             get_point(point, i, mask.n ); 
             double d = circle.center.distance(point) ; 
             bool inside = d - circle.radius < 0. ; 
             mask.a[i] = inside ; 
             if(inside) num_inside += 1 ; 
         } 
         return num_inside ; 
    }

    void dump_point( const char* msg, int i, int n ) const 
    {
        Point point ; 
        get_point(point, i, n ); 
        printf("%s  i %3d/%d   point %10.4f %10.4f \n", msg, i, n, point.x, point.y ); 
    } 

    void dump_points_first_last( const Mask& mask ) const 
    {
         int first = mask.first() ; 
         int last = mask.last() ; 

         dump_point("first", first, mask.n ); 
         dump_point("last ", last, mask.n ); 
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



struct Ellipse_Intersect_Circle
{
    Ellipse ellipse ; 
    Circle circle ; 
    Mask mask ; 

    int num_inside ; 
    bool verbose ; 

    Point first ; 
    Point last ; 

    static Ellipse_Intersect_Circle make(double e_cx, double e_cy, double e_ax, double e_ay,  double c_cx, double c_cy, double c_r, int n, bool verbose )
    {
        Ellipse_Intersect_Circle ec ; 
        ec.ellipse = Ellipse::make( e_cx, e_cy, e_ax, e_ay ); 
        ec.circle = Circle::make( c_cx, c_cy, c_r ); 
        ec.mask = Mask::make(n) ; 
        ec.verbose = verbose ; 
        int ni = ec.find_ellipse_points_inside_circle(); 
        assert( ni > 0); 
        return ec ; 
    }

    int find_ellipse_points_inside_circle()
    {
        num_inside = ellipse.points_inside_circle( mask, circle );  
        assert( num_inside > 0 ); 

        if(verbose)
        {
            printf(" num_inside %d \n", num_inside ); 
            mask.dump("ellipse points inside circle");  
            ellipse.dump_points_first_last( mask ); 
        }

        int fidx = ellipse.get_point_first( first, mask ); 
        

        if( fidx < 0 )
        {
            printf("no points on the ellipse are inside the circle\n");  
        } 
        else
        {
            if(verbose)
            printf(" fidx %d first (%10.4f, %10.4f) \n", fidx, first.x, first.y );  
        }

        int lidx = ellipse.get_point_last( last, mask ); 
        if( lidx < 0 )
        {
            printf("no points on the ellipse are inside the circle\n");  
        } 
        else
        {
            if(verbose)
            printf(" lidx %d last (%10.4f, %10.4f) \n", lidx, last.x, last.y );  
        }
        return num_inside ; 
    }
}; 



