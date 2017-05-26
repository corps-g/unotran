module material
  use control, only : xs_name, allow_fission

  implicit none

  integer :: number_materials, number_groups, debugFlag, number_legendre
  double precision, allocatable, dimension(:) :: ebounds, velocity
  double precision, allocatable, dimension(:,:) :: sig_t, sig_f, nu_sig_f, chi
  double precision, allocatable, dimension(:,:,:,:) :: sig_s

  contains

  ! Read the cross section data from the file
  subroutine create_material()
    ! Inputs :
    !   xs_name : file where cross sections are stored
    !   allow_fission : boolian for setting fission to zero or not

    ! Read a file that is stored in the proteus format
    character(256) :: materialName
    integer :: mat, g, gp, L, dataPresent
    double precision :: t, f, vf, c, energyFission, energyCapture, gramAtomWeight
    double precision, allocatable, dimension(:) :: array1
    
    ! Read the file parameters
    open(unit=5, file=xs_name)
    read(5,*) number_materials, number_groups, debugFlag
    allocate(ebounds(number_groups + 1))
    read(5,*) ebounds
    allocate(velocity(number_groups))
    read(5,*) velocity
    read(5,'(a)') materialName
    read(5,*) number_legendre, dataPresent, energyFission, energyCapture, gramAtomWeight
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
        read(5,'(a)') materialName
        read(5,*) number_legendre, dataPresent, energyFission, energyCapture, gramAtomWeight
        ! Count the highest order + zeroth order
        number_legendre = number_legendre - 1
      end if
      do g = 1, number_groups
        if (dataPresent == 1) then
          ! Read total and fission cross sections
          read(5,*) t, f, vf, c
          sig_t(g, mat) = t
          sig_f(g, mat) = f
          nu_sig_f(g, mat) = vf
          chi(g, mat) = c
        else
          ! Read just the total cross section
          read(5,*) t
          sig_t(g, mat) = t
          sig_f(g, mat) = 0.0
          nu_sig_f(g, mat) = 0.0
          chi(g, mat) = 0.0
        end if
      end do
      ! Read scattering cross section
      do l = 0, number_legendre
        do g = 1, number_groups
          read(5,*) array1
          sig_s(l, :, g, mat) = array1(:)
        end do
      end do
    end do

    close(unit=5)
    deallocate(array1)

    if (.not. allow_fission) then
      sig_f = 0.0
      nu_sig_f = 0.0
    end if
      
  end subroutine create_material

  subroutine finalize_material()
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
    

