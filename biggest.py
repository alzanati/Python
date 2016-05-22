def biggest(a, b, c):
    if( a < c ):
        if( b < c ):
            return c
        else:
            return b
    else:
        if( a > b ):
            return a
        else:
            return b
