module material
  implicit none
  integer :: number_materials, number_groups, debugFlag, number_legendre
  double precision, allocatable, dimension(:) :: ebounds, velocity
  double precision, allocatable, dimension(:,:) :: sig_t, sig_f, vsig_f, chi
  double precision, allocatable, dimension(:,:,:,:) :: sig_s

  contains

  ! Read the cross section data from the file
  subroutine create_material(fileName)
    ! Inputs :
    !   fileName : file where cross sections are stored

    ! Read a file that is stored in the proteus format
    character(len=*), intent(in) :: fileName
    character(1000) :: materialName
    integer :: mat, g, gp, L, dataPresent
    double precision :: t, f, vf, c, energyFission, energyCapture, gramAtomWeight
    double precision, allocatable, dimension(:) :: array1
    
    ! Read the file parameters
    open(unit=5, file=fileName)
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
    allocate(vsig_f(number_groups, number_materials))
    allocate(chi(number_groups, number_materials))
    allocate(sig_s(0:number_legendre, number_groups, number_groups, number_materials))
    allocate(array1(number_groups))
    
    ! Read the cross sections from the file
    do mat = 1, number_materials
      if (mat .gt. 1) then  ! The first material was read above to get array sizes
        read(5,'(a)') materialName
        read(5,*) number_legendre, dataPresent, energyFission, energyCapture, gramAtomWeight
        ! Count the highest order + zeroth order
        number_legendre = number_legendre - 1
      end if
      do g = 1, number_groups
        if (dataPresent .eq. 1) then
          ! Read total and fission cross sections
          read(5,*) t, f, vf, c
          sig_t(g, mat) = t
          sig_f(g, mat) = f
          vsig_f(g, mat) = vf
          chi(g, mat) = c
        else
          ! Read just the total cross section
          read(5,*) t
          sig_t(g, mat) = t
          sig_f(g, mat) = 0.0
          vsig_f(g, mat) = 0.0
          chi(g, mat) = 0.0
        end if
      end do
      ! Read scattering cross section
      do L = 0, number_legendre
        do g = 1, number_groups
          read(5,*) array1
          do gp = 1, number_groups
            sig_s(L, g, gp, mat) = array1(gp)
          end do
        end do
      end do
      
      ! make sure chi is a PDF
      if (dataPresent .eq. 1) then
        chi(:,mat) = chi(:,mat) / sum(chi(:,mat))
      end if
    end do
      
  end subroutine create_material

end module material
    

