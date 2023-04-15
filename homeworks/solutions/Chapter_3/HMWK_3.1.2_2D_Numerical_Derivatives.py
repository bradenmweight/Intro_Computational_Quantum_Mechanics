import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp

################# BEGIN USER INPUT #################
def get_globals():
    global x_GRID, y_GRID, Nx, Ny, DATA_DIR

    x_GRID = np.linspace( 0,3,1000 ) # START, END, NPOINTS   
    y_GRID = np.linspace( 0,3,1000 ) # START, END, NPOINTS   

    Nx = len( x_GRID )
    Ny = len( y_GRID )
    DATA_DIR = "3.1.1_PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}",shell=True)

def get_f_x_y():
    f_x_y = np.zeros(( Nx, Ny ))
    for xi in range( Nx ):
        for yi in range( Ny ):
            f_x_y[ xi, yi ] = x_GRID[ xi ] ** 2 + y_GRID[ yi ] ** 2
            

    return f_x_y

def get_NUM_f_xp_y( f_x_y ):

    # First-order finite difference (central) in first index (x)
    f_xp_y = np.zeros(( Nx, Ny ))
    dx = x_GRID[1] - x_GRID[0]
    for xi in range( Nx ):
        if ( xi == 0 ):
            # Do forward difference for the first point,
            #   since there are no points behind
            f_xp_y[xi,:]  = f_x_y[xi+1,:] - f_x_y[xi,:]
            f_xp_y[xi,:] /= dx
        if ( xi < Nx-1 ):
            f_xp_y[xi,:]  = f_x_y[xi+1,:] - f_x_y[xi-1,:]
            f_xp_y[xi,:] /= 2 * dx
        if ( xi == Nx-1 ):
            # Do backward difference for the last point,
            #   since there are no points ahead
            f_xp_y[xi,:]  = f_x_y[xi-1,:] - f_x_y[xi,:]
            f_xp_y[xi,:] /= dx
    return f_xp_y

def get_NUM_f_x_yp( f_x_y ):

    # First-order finite difference (central) in second index (y)
    f_x_yp = np.zeros(( Nx, Ny ))
    dy = y_GRID[1] - y_GRID[0]
    for yi in range( Ny ):
        if ( yi == 0 ):
            # Do forward difference for the first point,
            #   since there are no points behind
            f_x_yp[:,yi]  = f_x_y[:,yi+1] - f_x_y[:,yi]
            f_x_yp[:,yi] /= dy
        if ( yi < Ny-1 ):
            f_x_yp[:,yi]  = f_x_y[:,yi+1] - f_x_y[:,yi-1]
            f_x_yp[:,yi] /= 2 * dy
        if ( yi == Ny-1 ):
            # Do backward difference for the last point,
            #   since there are no points ahead
            f_x_yp[:,yi]  = f_x_y[:,yi-1] - f_x_y[:,yi]
            f_x_yp[:,yi] /= dy
    return f_x_yp


def get_NUM_f_xpp_y( f_x_y ):

    # First-order finite difference (central) in first index (x)
    f_xp_y = np.zeros(( Nx, Ny ))
    dx = x_GRID[1] - x_GRID[0]
    for xi in range( Nx ):
        if ( xi == 0 ):
            f_xp_y[xi,:] = 0
        if ( xi < Nx-1 ):
            f_xp_y[xi,:]  = f_x_y[xi+1,:] - 2 * f_x_y[xi,:] + f_x_y[xi-1,:]
            f_xp_y[xi,:] /= dx**2
        if ( xi == Nx-1 ):
            f_xp_y[xi,:] = 0

    return f_xp_y


def get_NUM_f_x_ypp( f_x_y ):

    # Second-order finite difference (central) in second index (y)
    f_x_yp = np.zeros(( Nx, Ny ))
    dy = y_GRID[1] - y_GRID[0]
    for yi in range( Ny ):
        if ( yi == 0 ):
            f_x_yp[:,yi]   = 0
        if ( yi < Ny-1 ):
            f_x_yp[:,yi]  = f_x_y[:,yi+1] - 2 * f_x_y[:,yi] + f_x_y[:,yi-1]
            f_x_yp[:,yi] /= dy ** 2
        if ( yi == Ny-1 ):
            f_x_yp[:,yi]   = 0
    return f_x_yp

def get_NUM_f_xp_yp( f_x_y ):

    # Second-order finite difference (central) in first (x) and second (y) index
    f_xp_yp = np.zeros(( Nx, Ny ))
    dx = x_GRID[1] - y_GRID[0]
    dy = y_GRID[1] - y_GRID[0]
    for xi in range( Nx ):
        for yi in range( Ny ):
            if ( xi == 0 ):
                f_xp_yp[xi,:]   = 0
            if ( yi == 0 ):
                f_xp_yp[:,yi]   = 0

            if ( xi < Nx-1 and yi < Ny-1 ):
                f_xp_yp[xi,yi]  +=     f_x_y[xi+1, yi+1]
                f_xp_yp[xi,yi]  -=     f_x_y[xi+1, yi  ]
                f_xp_yp[xi,yi]  -=     f_x_y[xi,   yi+1]
                f_xp_yp[xi,yi]  += 2 * f_x_y[xi,   yi  ]
                f_xp_yp[xi,yi]  -=     f_x_y[xi,   yi-1]
                f_xp_yp[xi,yi]  -=     f_x_y[xi-1, yi  ]
                f_xp_yp[xi,yi]  +=     f_x_y[xi-1, yi-1]

                f_xp_yp[xi,yi]  /= 2 * dx * dy

            if ( xi == Nx-1 ):
                f_xp_yp[xi,:]   = 0
            if ( yi == Ny-1 ):
                f_xp_yp[:,yi]   = 0
    return f_xp_yp


def plot_f_xp_y( f_x_y, f_xp_y ):

    IND_Y = 100
    print( "Y Index: ", IND_Y, y_GRID[IND_Y] )
    g    = f_xp_y[:,IND_Y] # df/dx ( x, y = 1 )
    f    = f_x_y[:,IND_Y] # This is the original function

    plt.plot(x_GRID[1:-1],f[1:-1],"-",c='black',lw=2,label="$f(x,y=1)$")
    plt.plot(x_GRID[1:-1],g[1:-1],"-",c='red',lw=2,label="$\\frac{\partial f(x,y)}{\partial x}|_{y=1}$ (Central Diff.)")
    plt.legend()
    plt.xlim(x_GRID[1],x_GRID[-1])
    plt.savefig(f"{DATA_DIR}/g1.jpg",dpi=300)
    plt.clf()


def plot_f_x_yp(  f_x_y, f_x_yp ):

    IND_X = 100
    print( "X Index: ", IND_X, x_GRID[IND_X] )
    g    = f_x_yp[IND_X,:] # df/dx ( x, y = 1 )
    f    = f_x_y[IND_X,:] # This is the original function

    plt.plot(y_GRID[1:-1],f[1:-1],"-",c='black',lw=2,label="$f(x=1,y)$")
    plt.plot(y_GRID[1:-1],g[1:-1],"-",c='red',lw=2,label="$\\frac{\partial f(x,y)}{\partial y}|_{x=1}$ (Central Diff.)")
    plt.legend()
    plt.xlim(y_GRID[1],y_GRID[-1])
    plt.savefig(f"{DATA_DIR}/g2.jpg",dpi=300)
    plt.clf()

def plot_f_xpp_y(  f_x_y, f_xpp_y ):

    IND_Y = 100
    print( "Y Index: ", IND_Y, y_GRID[IND_Y] )
    g    = f_xpp_y[:,IND_Y] # df/dx ( x, y = 1 )
    f    = f_x_y[:,IND_Y] # This is the original function

    plt.plot(x_GRID[1:-1],f[1:-1],"-",c='black',lw=2,label="$f(x,y=1)$")
    plt.plot(x_GRID[1:-1],g[1:-1],"-",c='red',lw=2,label="$\\frac{\partial^2 f(x,y)}{\partial x^2}|_{y=1}$ (Central Diff.)")
    plt.legend()
    plt.xlim(x_GRID[1],x_GRID[-1])
    plt.savefig(f"{DATA_DIR}/g3.jpg",dpi=300)
    plt.clf()

def plot_f_x_ypp(  f_x_y, f_x_ypp ):

    IND_X = 100
    print( "X Index: ", IND_X, x_GRID[IND_X] )
    g    = f_x_ypp[:,IND_X] # df/dx ( x, y = 1 )
    f    = f_x_y[IND_X,:] # This is the original function

    plt.plot(y_GRID[1:-1],f[1:-1],"-",c='black',lw=2,label="$f(x=1,y)$")
    plt.plot(y_GRID[1:-1],g[1:-1],"-",c='red',lw=2,label="$\\frac{\partial^2 f(x,y)}{\partial y^2}|_{x=1}$ (Central Diff.)")
    plt.legend()
    plt.xlim(y_GRID[1],y_GRID[-1])
    plt.savefig(f"{DATA_DIR}/g4.jpg",dpi=300)
    plt.clf()

def plot_f_xp_yp(  f_x_y, f_xp_yp ):

    IND_Y = 100
    print( "Y Index: ", IND_Y, y_GRID[IND_Y] )
    g    = f_xp_yp[:,IND_Y] # df/dx ( x, y = 1 )
    f    = f_x_y[:,IND_Y] # This is the original function

    plt.plot(x_GRID[1:-1],f[1:-1],"-",c='black',lw=2,label="$f(x=1,y)$")
    plt.plot(x_GRID[1:-1],g[1:-1],"-",c='red',lw=2,label="$\\frac{\partial^2 f(x,y)}{\partial x \partial y}|_{y=1}$ (Central Diff.)")
    plt.legend()
    plt.xlim(x_GRID[1],x_GRID[-1])
    plt.savefig(f"{DATA_DIR}/g5.jpg",dpi=300)
    plt.clf()


def main():
    get_globals()
    f_x_y  = get_f_x_y()

    f_xp_y  = get_NUM_f_xp_y( f_x_y )
    f_x_yp  = get_NUM_f_x_yp( f_x_y )
    f_xpp_y = get_NUM_f_xpp_y( f_x_y )
    f_x_ypp = get_NUM_f_x_ypp( f_x_y )
    f_xp_yp = get_NUM_f_xp_yp( f_x_y )

    plot_f_xp_y (f_x_y, f_xp_y )
    plot_f_x_yp (f_x_y, f_x_yp )
    plot_f_xpp_y(f_x_y, f_xpp_y )
    plot_f_x_ypp(f_x_y, f_xpp_y )
    plot_f_xp_yp(f_x_y, f_xp_yp )

if ( __name__ == "__main__" ):
    main()




