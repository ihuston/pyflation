subroutine rk4stepks(x, y, h, n, m, yout, derivs)
!   Do one step of the classical 4th order Runge Kutta method,
!   starting from y at x with time step h and derivatives given by derivs
    implicit none

    real, intent(in) :: x
    real, intent(in) :: h
    integer :: n, m
    real, intent(in) :: y(0:n-1,0:m-1)


    real :: hh, h6, xh
    real, dimension(0:n-1,0:m-1) :: yt, dyt, dym, dydx
    real :: derivs
    real, dimension(0:n-1,0:m-1), intent(out) :: yout
    external derivs
!f2py real, intent(in), dimension(0:n-1,0:m-1) :: y
!f2py real, intent(out), dimension(0:n-1,0:m-1) :: yout
!f2py integer, intent(in) :: n, m
!f2py real, intent(in) :: x, h
!f2py depend(n,m) :: y, yt, dyt, dym, yout, derivs

    hh = h*0.5 !Half time step
    h6 = h/6.0 !Sixth of time step
    xh = x + hh ! Halfway point in x direction

    dydx = derivs(y,x)

    !First step, we already have derivatives from dydx
    !yt = y + hh*dydx

    !Second step, get new derivatives
    !dyt = derivs(yt, xh)

    !yt = y + hh*dyt

    !Third step
    !dym = derivs(yt, xh)

    !yt = y + h*dym
    !dym = dym + dyt

    !Fourth step
    !dyt = derivs(yt, x+h)

    !Accumulate increments with proper weights
    !yout = y + h6*(dydx + dyt + 2*dym)
    yout = dydx

    return
    end subroutine
