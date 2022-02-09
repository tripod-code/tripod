subroutine calculate_a(smin, smax, sint, xi, a, Nr, Nm)
  ! Subroutine calculates the particle sizes.
  ! a = [a0, a1, 0.5*amean, amean]
  !
  ! Parameters
  ! ----------
  ! smin(Nr) : Minimal particle size
  ! smax(Nr) : Maximum particle size
  ! sint(Nr) : Intermediate particle size
  ! xi(Nr) : Calculated distribution exponent
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! a(Nr, Nm) : Particle sizes

  implicit None

  double precision, intent(in)  :: smin(Nr)
  double precision, intent(in)  :: smax(Nr)
  double precision, intent(in)  :: sint(Nr)
  double precision, intent(in)  :: xi(Nr)
  double precision, intent(out) :: a(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i

  do i=1, Nr
    if(xi(i) .eq. -5) then
      a(i, 1) = &
      sint(i) * smin(i) / ( sint(i) - smin(i) ) * log( sint(i) / smin(i) )
      a(i, 2) = &
      smax(i) * sint(i) / ( smax(i) - sint(i) ) * log( smax(i) / sint(i) )
      a(i, 4) = &
      smax(i) * smin(i) / ( smax(i) - smin(i) ) * log( smax(i) / smin(i) )
      a(i, 3) = 0.5d0 * a(i, 4)
    else if(xi(i) .eq. -4) then
      a(i, 1) = ( sint(i) - smin(i) ) / log( sint(i) / smin(i) )
      a(i, 2) = ( smax(i) - sint(i) ) / log( smax(i) / sint(i) )
      a(i, 4) = ( smax(i) - smin(i) ) / log( smax(i) / smin(i) )
      a(i, 3) = 0.5d0 * a(i, 4)
    else
      a(i, 1) = &
      ( xi(i) + 4.d0 ) / ( xi(i) + 5.d0 ) * ( sint(i)**( xi(i) + 5.d0 ) -&
      smin(i)**( xi(i) + 5.d0) ) / ( sint(i)**( xi(i) + 4.d0 ) - smin(i)&
      **( xi(i) + 4.d0) )
      a(i, 2) = &
      ( xi(i) + 4.d0 ) / ( xi(i) + 5.d0 ) * ( smax(i)**( xi(i) + 5.d0 ) -&
      sint(i)**( xi(i) + 5.d0) ) / ( smax(i)**( xi(i) + 4.d0 ) - sint(i)&
      **( xi(i) + 4.d0) )
      a(i, 4) = &
      ( xi(i) + 4.d0 ) / ( xi(i) + 5.d0 ) * ( smax(i)**( xi(i) + 5.d0 ) -&
      smin(i)**( xi(i) + 5.d0) ) / ( smax(i)**( xi(i) + 4.d0 ) - smin(i)&
      **( xi(i) + 4.d0) )
      a(i, 3) = 0.5d0 * a(i, 4)
    end if
  end do

end subroutine calculate_a


subroutine fi_adv(Sigma, v, r, ri, Fi, Nr, Nm_s, Nm_l)
  ! Function calculates the advective mass fluxes through the grid cell interfaces.
  ! Velocity v is linearly interpolated on grid cell interfaces with
  ! vi(1, :) = vi(2, :) and vi(Nr+1, :) = vi(Nr, :).
  !
  ! Parameters
  ! ----------
  ! Sigma(Nr, Nm) : Surface density
  ! v(Nr, Nm) : Radial velocity at grid cell centers
  ! r(Nr) : Radial grid cell centers
  ! ri(Nr+1) : Radial grid cell interfaces
  ! Nr : Number of radial grid cells
  ! Nm_s : Short number of mass bins
  ! Nm_l : Long number of mass bins
  !
  ! Returns
  ! -------
  ! Fi(Nr+1, Nm) : Flux through grid cell interfaces.

  implicit none

  double precision, intent(in)  :: Sigma(Nr, Nm_s)
  double precision, intent(in)  :: v(Nr, Nm_l)
  double precision, intent(in)  :: r(Nr)
  double precision, intent(in)  :: ri(Nr+1)
  double precision, intent(out) :: Fi(Nr+1, Nm_s)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm_s
  integer,          intent(in)  :: Nm_l

  double precision :: vi(Nr+1)
  integer :: i
  integer :: ir

  do i=1, Nm_s
    call interp1d(ri, r, v(:, i), vi, Nr)
    do ir=2, Nr
      Fi(ir, i) = Sigma(ir-1, i)*max(0.0, vi(ir)) + Sigma(ir, i)*min(vi(ir), 0.d0)
    end do
    Fi(1, i) = Sigma(1, i)*min(vi(2), 0.d0)
    Fi(Nr+1, i) = Sigma(Nr, i)*max(0.0, vi(Nr))
  end do

end subroutine fi_adv


subroutine fi_diff(D, SigmaD, SigmaG, St, u, r, ri, Fi, Nr, Nm_s, Nm_l)
  ! Subroutine calculates the diffusive dust fluxes at the grid cell interfaces.
  ! The flux at the boundaries is assumed to be constant.
  !
  ! Parameters
  ! ----------
  ! D(Nr, Nm) : Dust diffusivity
  ! SigmaD(Nr, Nm) : Dust surface densities
  ! SigmaG(Nr) : Gas surface density
  ! St(Nr, Nm) : Stokes number
  ! u(Nr) : Gas turbulent RMS velocity
  ! r(Nr) : Radial grid cell centers
  ! ri(Nr+1) : Radial grid cell interfaces
  ! Nr : Number of radial grid cells
  ! Nm_s : Short number of mass bins
  ! Nm_l : Long number of mass bins
  !
  ! Returns
  ! -------
  ! Fi(Nr+1, Nm) : Diffusive fluxes at grid cell interfaces

  implicit none

  double precision, intent(in)  :: D(Nr, Nm_l)
  double precision, intent(in)  :: SigmaD(Nr, Nm_s)
  double precision, intent(in)  :: SigmaG(Nr)
  double precision, intent(in)  :: St(Nr, Nm_l)
  double precision, intent(in)  :: u(Nr)
  double precision, intent(in)  :: r(Nr)
  double precision, intent(in)  :: ri(Nr+1)
  double precision, intent(out) :: Fi(Nr+1, Nm_s)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm_s
  integer,          intent(in)  :: Nm_l

  double precision :: Di(Nr+1, Nm_s)
  double precision :: eps(Nr, Nm_s)
  double precision :: gradepsi(Nr+1, Nm_s)
  double precision :: lambda
  double precision :: P
  double precision :: SigDi(Nr+1, Nm_s)
  double precision :: SigGi(Nr+1)
  double precision :: Sti(Nr+1, Nm_s)
  double precision :: ui(Nr+1)
  double precision :: w
  integer :: ir
  integer :: i

  Fi(:, :) = 0.d0

  call interp1d(ri, r, SigmaG, SigGi, Nr)
  call interp1d(ri, r, u, ui, Nr)

  do i=1, Nm_s
    call interp1d(ri(:), r(:), D(:, i), Di(:, i), Nr)
    call interp1d(ri(:), r(:), SigmaD(:, i), SigDi(:, i), Nr)
    eps(:, i) = SigmaD(:, i) / SigmaG(:)
    call interp1d(ri(:), r(:), St(:, i), Sti(:, i), Nr)
  end do

  do ir=2, Nr
    gradepsi(ir, :) = ( eps(ir, :) - eps(ir-1, :) ) / ( r(ir) - r(ir-1) )
  end do

  do i=1, Nm_s
    do ir=2, Nr
      w = ui(ir) * SigDi(ir, i) / (1.d0 + Sti(ir, i)**2)
      Fi(ir, i) = -Di(ir, i) * SigGi(ir) * gradepsi(ir, i)
      P = abs( Fi(ir, i) / w )
      lambda = ( 1.d0 + P ) / ( 1.d0 + P + P**2 )
      if(lambda .GT. HUGE(lambda)) then
        Fi(ir, i) = w
      else
        Fi(ir, i) = lambda * Fi(ir, i)
      end if
    end do
  end do

  Fi(   1, :) = Fi( 2, :)
  Fi(Nr+1, :) = Fi(Nr, :)

end subroutine fi_diff


subroutine m(a, rho, fill, masses, Nr, Nm)
  ! Subroutine calculates the particle masses.
  !
  ! Parameters
  ! ----------
  ! a(Nr, Nm) : Particle sizes
  ! rho(Nr, Nm) : Particle bulk densities
  ! fill(Nr, Nm) : Filling factor
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! masses(Nr, Nm) : Particle masses

  use constants, only: pi

  implicit None

  double precision, intent(in)  :: a(Nr, Nm)
  double precision, intent(in)  :: rho(Nr, Nm)
  double precision, intent(in)  :: fill(Nr, Nm)
  double precision, intent(out) :: masses(Nr, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i, j

  do i=1, Nr
    do j=1, Nm
      masses(i, j) = 4.d0 / 3.d0 * pi * rho(i, j) * fill(i, j) * a(i, j)**3.d0
    end do
  end do

end subroutine m


subroutine pfrag(vrel, vfrag, pf, Nr, Nm)
  ! Subroutine calculates the fragmentation probability.
  ! It is assuming a Maxwell-Boltzmann velocity distribution.
  ! 
  ! Parameters
  ! ----------
  ! vrel(Nr, Nm, Nm) : Relative velocity
  ! vfrag(Nr) : Fragmentation velocity
  ! Nr : Number or radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! pf(Nr, Nm, Nm) : Fragmentation probability in [0, 1]
  !
  ! Notes
  ! -----
  ! The sticking probability is ps = 1 - pf

  implicit none

  double precision, intent(in)  :: vrel(Nr, Nm, Nm)
  double precision, intent(in)  :: vfrag(Nr)
  double precision, intent(out) :: pf(Nr, Nm, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm
  
  double precision :: dum
  integer :: ir
  integer :: i
  integer :: j

  do i=1, Nm
    do j=1, i
      do ir=2, Nr-1
        dum = (vfrag(ir)/vrel(ir, j, i))**2
        pf(ir, j, i) = (1.5d0*dum + 1.d0) * exp(-1.5d0*dum)
        pf(ir, i, j) = pf(ir, j, i)
      end do
    end do
  end do

end subroutine pfrag


subroutine vrel_brownian_motion(cs, m, T, vrel, Nr, Nm)
  ! Subroutine calculates the relative particle velocities due to Brownian motion.
  ! Its maximum value is the sound speed.
  !
  ! Parameters
  ! ----------
  ! cs(Nr) : Sound speed
  ! m(Nr, Nm) : Particle masses
  ! T(Nr) : Temperature
  ! Nr : Number of radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! vrel(Nr, Nm, Nm) : Relative velocities

  use constants, only: k_B, pi

  implicit none

  double precision, intent(in)  :: cs(Nr)
  double precision, intent(in)  :: m(Nr, Nm)
  double precision, intent(in)  :: T(Nr)
  double precision, intent(out) :: vrel(Nr, Nm, Nm)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: ir, i, j
  double precision :: fac1
  double precision :: fac2(Nr)
  double precision :: dum

  fac1 = 8.d0 * k_B / pi

  do ir=1, Nr
    fac2(ir) = fac1 * T(ir)
  end do

  do i=1, Nm
    do j=1, i
      do ir=1, Nr
        dum = min(sqrt(fac2(ir)*(m(ir, j)+m(ir, i))/(m(ir, j)*m(ir, i))), cs(ir))
        vrel(ir, j, i) = dum
        vrel(ir, i, j) = dum
      end do
    end do
  end do

end subroutine vrel_brownian_motion


subroutine calculate_xi(Sigma, smax, sint, xi, Nr, Nm)
  ! Subroutine calculates the particle size distribution exponent.
  !
  ! Parameters
  ! ----------
  ! Sigma (Nr, Nm) : Dust surface density
  ! smax(Nr) : Maximum particle size
  ! sint(Nr) : Intermediate particle size
  ! Nr : Number or radial grid cells
  ! Nm : Number of mass bins
  !
  ! Returns
  ! -------
  ! xi(Nr) : Calculated distribution exponent

  implicit none

  double precision, intent(in)  :: Sigma(Nr, Nm)
  double precision, intent(in)  :: smax(Nr)
  double precision, intent(in)  :: sint(Nr)
  double precision, intent(out) :: xi(Nr)
  integer,          intent(in)  :: Nr
  integer,          intent(in)  :: Nm

  integer :: i

  do i=1, Nr
    xi(i) = log( Sigma(i, 1) / Sigma(i, 2) ) / log( smax(i) / sint(i) ) - 4.d0
  end do

end subroutine calculate_xi


subroutine calculate_sint(smin, smax, sint, Nr)
  ! Subroutine calculates the intermediate particle size.
  !
  ! Parameters
  ! ----------
  ! smin(Nr) : Minimum particle size
  ! smax(Nr) : Maximum particle size
  ! Nr : Number or radial grid cells
  !
  ! Returns
  ! -------
  ! sint(Nr) : Intermediate particle size

  implicit none

  double precision, intent(in)  :: smin(Nr)
  double precision, intent(in)  :: smax(Nr)
  double precision, intent(out) :: sint(Nr)
  integer,          intent(in)  :: Nr

  sint = sqrt( smin * smax )

end subroutine calculate_sint
