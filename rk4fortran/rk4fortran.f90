subroutine rk4stepks(x, y, h, n, m, yout)
!   Do one step of the classical 4th order Runge Kutta method,
!   starting from y at x with time step h and derivatives given by derivs
    implicit none

    real, intent(in) :: x
    real, intent(in) :: y
    real, intent(in) :: h
    integer, intent(in) :: n
    integer, intent(in) :: m
    !external derivs

    real :: hh, h6, xh
    real, dimension(0:n-1,0:m-1) :: yt, dyt, dym
    real, dimension(0:n-1,0:m-1), intent(out) :: yout


    hh = h*0.5 !Half time step
    h6 = h/6.0 !Sixth of time step
    xh = x + hh ! Halfway point in x direction

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

    return
