// name=union2Test


/**
https://stackoverflow.com/questions/42856717/intrinsics-equivalent-to-the-cuda-type-casting-intrinsics-double2loint-doub

**/


double hiloint2double(int hi, int lo)
{
    union {
        double val;
        struct {
            int lo;
            int hi;
        };
    } u;
    u.hi = hi;
    u.lo = lo;
    return u.val;
}

int double2hiint(double val)
{
    union {
        double val;
        struct {
            int lo;
            int hi;
        };
    } u;
    u.val = val;
    return u.hi;
}

int double2loint(double val)
{
    union {
        double val;
        struct {
            int lo;
            int hi;
        };
    } u;
    u.val = val;
    return u.lo;
}




