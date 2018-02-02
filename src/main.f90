program main
  use control, only : initialize_control, store_psi
  use dgmsolver
  use solver

  implicit none
  
  ! initialize types
  character(80)  :: inputfile
  integer :: l, c, a, g, n
  ! fineMesh : vector of int for number of fine mesh divisions per cell
  ! materialMap : vector of int with material index for each coarse region
  integer, allocatable :: fm(:), mm(:), em(:)
  ! coarseMap : vector of float with bounds for coarse mesh regions
  double precision, allocatable :: cm(:)
  double precision :: boundary(2)
  
  if ( COMMAND_ARGUMENT_COUNT() .lt. 1 ) then
    stop "*** ERROR: user input file not specified ***"
  else
    call get_command_argument(1, inputfile);
  end if

  call initialize_control(inputfile)

  if (use_DGM) then
    call initialize_dgmsolver()
    call dgmsolve()
    call dgmoutput()
    call finalize_dgmsolver()
  else
    call initialize_solver()
    call solve()
    call output()
    call finalize_solver()
  end if

  call finalize_control()

end program main
