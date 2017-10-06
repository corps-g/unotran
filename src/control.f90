module control
  implicit none

  ! control variables
  double precision, allocatable :: coarse_mesh(:)
  integer, allocatable :: fine_mesh(:), material_map(:), energy_group_map(:), truncation_map(:)
  double precision :: boundary_type(2), outer_tolerance, inner_tolerance, lambda=1.0, source_value=0.0
  character(:), allocatable :: xs_name, dgm_basis_name, equation_type, file_name, initial_phi, initial_psi, solver_type
  integer :: angle_order, angle_option, dgm_expansion_order=-1, legendre_order=-1
  logical :: allow_fission=.false., outer_print=.true., inner_print=.false.
  logical :: use_dgm=.false., store_psi=.false., use_recondensation=.false.

  contains

  subroutine initialize_control(fname, silent)
    ! Input variables
    character(len=256) :: buffer, label
    character(len=*), intent(in) :: fname
    integer :: pos
    integer, parameter :: fh = 15
    integer :: ios = 0
    integer :: line = 0
    logical, optional :: silent
    logical :: no_print

    if (present(silent)) then
      no_print = silent
    else
      no_print = .false.
    end if

    file_name = trim(adjustl(fname))

    open(fh, file=file_name, action='read', iostat=ios)
    if (ios > 0) stop "*** ERROR: user input file not found ***"

    ! ios is negative if an end of record condition is encountered or if
    ! an endfile condition was detected.  It is positive if an error was
    ! detected.  ios is zero otherwise.
    do while (ios == 0)
      read(fh, '(A)', iostat=ios) buffer
      if (ios == 0) then
        line = line + 1

        ! Find the first whitespace and split label and data
        pos = scan(buffer, '    ')
        label = buffer(1:pos)
        buffer = buffer(pos+1:)

        select case (label)
        case ('fine_mesh')
          allocate(fine_mesh(nitems(buffer)))
          read(buffer, *, iostat=ios) fine_mesh
        case ('coarse_mesh')
          allocate(coarse_mesh(nitems(buffer)))
          read(buffer, *, iostat=ios) coarse_mesh
        case ('material_map')
          allocate(material_map(nitems(buffer)))
          read(buffer, *, iostat=ios) material_map
        case ('xs_file')
          xs_name=trim(adjustl(buffer))
        case ('initial_phi')
          initial_phi=trim(adjustl(buffer))
        case ('initial_psi')
          initial_psi=trim(adjustl(buffer))
        case ('angle_order')
          read(buffer, *, iostat=ios) angle_order
        case ('angle_option')
          read(buffer, *, iostat=ios) angle_option
        case ('boundary_type')
          read(buffer, *, iostat=ios) boundary_type
        case ('allow_fission')
          read(buffer, *, iostat=ios) allow_fission
        case ('energy_group_map')
          allocate(energy_group_map(nitems(buffer)))
          read(buffer, *, iostat=ios) energy_group_map
        case ('dgm_basis_file')
          dgm_basis_name=trim(adjustl(buffer))
        case ('truncation_map')
          allocate(truncation_map(nitems(buffer)))
          read(buffer, *, iostat=ios) truncation_map
        case ('outer_print')
          read(buffer, *, iostat=ios) outer_print
        case ('inner_print')
          read(buffer, *, iostat=ios) inner_print
        case ('outer_tolerance')
          read(buffer, *, iostat=ios) outer_tolerance
        case ('inner_tolerance')
          read(buffer, *, iostat=ios) inner_tolerance
        case ('lambda')
          read(buffer, *, iostat=ios) lambda
        case ('use_DGM')
          read(buffer, *, iostat=ios) use_DGM
        case ('store_psi')
          read(buffer, *, iostat=ios) store_psi
        case ('use_recondensation')
          read(buffer, *, iostat=ios) use_recondensation
        case ('equation_type')
          equation_type=trim(adjustl(buffer))
        case ('solver_type')
          solver_type=trim(adjustl(buffer))
        case ('source')
          read(buffer, *, iostat=ios) source_value
        case ('legendre_order')
          read(buffer, *, iostat=ios) legendre_order
        case default
          print *, 'Skipping invalid label at line', line
        end select
      end if
    end do
    close(fh)

    if (.not. allocated(equation_type)) then
      equation_type = 'DD'
    end if

    if (use_DGM) then
      store_psi = .true.
    end if

    if (.not. no_print) then
      call output_control()
    end if

    ! Kill the program if an invalid solver_type is selected
    if (solver_type == 'eigen') then
      continue
    else if (solver_type == 'fixed') then
      continue
    else
      print *, 'FATAL ERROR : Invalid solver type'
      stop
    end if

  end subroutine initialize_control

  subroutine output_control()

    print *, 'MESH VARIABLES'
    print *, '  fine_mesh          = [', fine_mesh, ']'
    print *, '  coarse_mesh        = [', coarse_mesh, ']'
    print *, '  material_map       = [', material_map, ']'
    print *, '  boundary_type      = [', boundary_type, ']'
    print *, 'MATERIAL VARIABLES'
    print *, '  xs_file_name       = "', xs_name, '"'
    print *, 'ANGLE VARIABLES'
    print *, '  angle_order        = ', angle_order
    print *, '  angle_option       = ', angle_option
    print *, 'SOURCE'
    print *, '  constant source    = ', source_value
    print *, 'OPTIONS'
    print *, '  solver_type        = "', solver_type, '"'
    print *, '  equation_type      = "', equation_type, '"'
    print *, '  store_psi          = ', store_psi
    print *, '  allow_fission      = ', allow_fission
    print *, '  outer_print        = ', outer_print
    print *, '  inner_print        = ', inner_print
    print *, '  outer_tolerance    = ', outer_tolerance
    print *, '  inner_tolerance    = ', inner_tolerance
    print *, '  lambda             = ', lambda
    if (legendre_order > -1) then
      print *, '  legendre_order     = ', legendre_order
    else
      print *, '  legendre_order     = DEFAULT'
    end if
    if (use_DGM) then
      print *, 'DGM OPTIONS'
      print *, '  dgm_basis_file     = "', dgm_basis_name, '"'
      print *, '  use_DGM            = ', use_DGM
      print *, '  use_recondensation = ', use_recondensation
      print *, '  energy_group_map   = [', energy_group_map, ']'
      if (allocated(truncation_map)) then
        print *, '  truncation_map     = [', truncation_map, ']'
      end if
    else
      print *, '  use_dgm            = ', use_dgm
    end if

  end subroutine output_control

  subroutine finalize_control()
    if (allocated(fine_mesh)) then
      deallocate(fine_mesh)
    end if
    if (allocated(coarse_mesh)) then
      deallocate(coarse_mesh)
    end if
    if (allocated(material_map)) then
      deallocate(material_map)
    end if
    if (allocated(energy_group_map)) then
      deallocate(energy_group_map)
    end if
    if (allocated(truncation_map)) then
      deallocate(truncation_map)
    end if
  end subroutine finalize_control

  ! number of space-separated items in a line
  integer function nitems(line)
    character line*(*)
    logical back
    integer length, k

    back = .true.
    length = len_trim(line)
    k = index(line(1:length), ' ', back)
    if (k == 0) then
      nitems = 0
      return
    end if

    nitems = 1
    do
      ! starting with the right most blank space,
      ! look for the next non-space character down
      ! indicating there is another item in the line
      do
        if (k <= 0) exit
        if (line(k:k) == ' ') then
          k = k - 1
          cycle
        else
          nitems = nitems + 1
          exit
        end if
      end do

      ! once a non-space character is found,
      ! skip all adjacent non-space character
      do
        if ( k<=0 ) exit
        if (line(k:k) /= ' ') then
          k = k - 1
          cycle
        end if
        exit
      end do
      if (k <= 0) exit
    end do
  end function nitems

end module control
