module material
  ! ############################################################################
  ! Initialize the material properties
  ! ############################################################################

  implicit none

  integer :: &
      number_materials, & ! Number of materials in the cross section library
      debugFlag           ! Unused flag in the cross section library
  double precision, allocatable, dimension(:) :: &
      ebounds,          & ! Bounds for the energy groups
      velocity            ! Velocity within each energy group
  double precision, allocatable, dimension(:,:) :: &
      sig_t,            & ! Total cross section
      sig_f,            & ! Fission cross section
      nu_sig_f,         & ! Fission cross section times nu
      chi                 ! Chi spectrum
  double precision, allocatable, dimension(:,:,:,:) :: &
      sig_s               ! Scattering cross section

  contains

  subroutine create_material()
    ! ##########################################################################
    ! Read the cross section data from a file in the proteus format
    ! ##########################################################################

    ! Use Statements
    use control, only : allow_fission, legendre_order, number_fine_groups, &
                        number_legendre, xs_name, number_coarse_groups

    ! Variable definitions
    character(256) :: &
        materialName     ! Name of each material
    integer :: &
        mat,           & ! Material index
        g,             & ! Outer group index
        L,             & ! Legendre moment index
        number_groups, & ! Number of groups in the cross section library
        dataPresent      ! Flag deciding which cross sections are present
    double precision :: &
        t,             & ! Total cross section value
        f,             & ! fission cross section value
        vf,            & ! nu times fission cross section value
        c,             & ! chi spectrum value
        energyFission, & ! Energy per fission event (unused)
        energyCapture, & ! Energy per capture event (unused)
        gramAtomWeight   ! Atomic weight
    double precision, allocatable, dimension(:) :: &
        array1           ! Temp array for storing scattering XS values
    
    ! Read the file parameters
    open(unit=1, file=xs_name)
    read(1,*) number_materials, number_groups, debugFlag
    number_fine_groups = number_groups
    number_coarse_groups = number_groups
    allocate(ebounds(number_groups + 1))
    read(1,*) ebounds
    allocate(velocity(number_groups))
    read(1,*) velocity
    read(1,'(a)') materialName
    read(1,*) number_legendre, dataPresent, energyFission, energyCapture, gramAtomWeight
    ! Count the highest order + zeroth order
    number_legendre = number_legendre - 1

    ! Make space for cross sections
    allocate(sig_t(number_groups, number_materials))
    allocate(sig_f(number_groups, number_materials))
    allocate(nu_sig_f(number_groups, number_materials))
    allocate(chi(number_groups, number_materials))
    allocate(sig_s(0:number_legendre, number_groups, number_groups, number_materials))
    allocate(array1(number_groups))

    ! Read the cross sections from the file
    do mat = 1, number_materials
      if (mat > 1) then  ! The first material was read above to get array sizes
        read(1,'(a)') materialName
        read(1,*) number_legendre, dataPresent, energyFission, energyCapture, gramAtomWeight
        ! Count the highest order + zeroth order
        number_legendre = number_legendre - 1
      end if
      do g = 1, number_groups
        if (dataPresent == 1) then
          ! Read total and fission cross sections
          read(1,*) t, f, vf, c
          sig_t(g, mat) = t
          sig_f(g, mat) = f
          nu_sig_f(g, mat) = vf
          chi(g, mat) = c
        else
          ! Read just the total cross section
          read(1,*) t
          sig_t(g, mat) = t
          sig_f(g, mat) = 0.0
          nu_sig_f(g, mat) = 0.0
          chi(g, mat) = 0.0
        end if
      end do
      ! Read scattering cross section
      do l = 0, number_legendre
        do g = 1, number_groups
          read(1,*) array1
          sig_s(l, g, :, mat) = array1(:)
        end do
      end do
    end do

    ! Close the file and clean up
    close(unit=1)
    deallocate(array1)

    ! If fissioning is not allowed, set fission cross sections to zero
    if (.not. allow_fission) then
      sig_f = 0.0
      nu_sig_f = 0.0
    end if

    ! Adjust the legendre order if given in control
    if (legendre_order /= -1) then
      number_legendre = legendre_order
    end if

  end subroutine create_material

  subroutine finalize_material()
    ! ##########################################################################
    ! Deallocate all used arrays
    ! ##########################################################################

    if (allocated(ebounds)) then
      deallocate(ebounds)
    end if
    if (allocated(velocity)) then
      deallocate(velocity)
    end if
    if (allocated(sig_t)) then
      deallocate(sig_t)
    end if
    if (allocated(sig_f)) then
      deallocate(sig_f)
    end if
    if (allocated(chi)) then
      deallocate(chi)
    end if
    if (allocated(nu_sig_f)) then
      deallocate(nu_sig_f)
    end if
    if (allocated(sig_s)) then
      deallocate(sig_s)
    end if
  end subroutine finalize_material

end module material
    

