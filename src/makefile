# Makefile for linear algebra tests 

#===============================================================================
# User Options
#===============================================================================

COMPILER = gfortran
OPTIMIZE = yes

#===============================================================================
# Vendor BLAS implementations
#===============================================================================

# For linking MKL, see:
#   http://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
LIBMKLSERIAL   = -I$(MKLROOT)/include -Wl,--start-group       \
                 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a   \
                 $(MKLROOT)/lib/intel64/libmkl_sequential.a   \
                 $(MKLROOT)/lib/intel64/libmkl_core.a         \
                 -Wl,--end-group -lpthread -lm
LIBMKLTHREAD   = -I$(MKLROOT)/include  -Wl,--start-group      \
                 $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a   \
                 $(MKLROOT)/lib/intel64/libmkl_intel_thread.a \
                 $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,    \
                 --end-group -openmp -lpthread -lm

# BLAS defines a generic set of functions with vendor-specific
# implementations; here, we use the blas implementation from the 
# software repository.
LIBBLAS        = -L/usr/lib -lblas -llapack

#===============================================================================
# Compiler Definitions
#===============================================================================

ifeq ($(COMPILER),gfortran)
  F90 = gfortran
  OMPFLAG = -fopenmp
  ifeq ($(OPTIMIZE),yes)
    OPT = -O3
  endif
endif

ifeq ($(COMPILER),ifort)
  F90 = ifort
  OMPFLAG = -openmp
  ifeq ($(OPTIMIZE),yes)
    OPT = -O3
  endif
endif

#===============================================================================
# Targets
#===============================================================================

# Use the repository's BLAS implementation
angle:
	f2py2.7 --fcompiler=$(COMPILER) $(LIBBLAS) -c angle.f90 -m $@ 

mesh:
	f2py2.7 --fcompiler=$(COMPILER) $(LIBBLAS) -c mesh.f90 -m $@ 
	
material:
	f2py2.7 --fcompiler=$(COMPILER) $(LIBBLAS) -c material.f90 -m $@
	
state:
	f2py2.7 --fcompiler=$(COMPILER) $(LIBBLAS) -c material.f90 mesh.f90 state.f90 -m $@
	
sweeper:
	f2py2.7 --fcompiler=$(COMPILER) $(LIBBLAS) -c sweeper.f90 -m $@

main:
	$(COMPILER) $(OMPFLAG) $(LIBBLAS) -o main.exe material.f90 mesh.f90 angle.f90 state.f90 main.f90

# Remove objects, module files, and exectables
clean:
	@rm -f *.o *.mod *-x *wrap.cxx *.so

# Same, but leave the executable
neat:
	@rm -f *.o *.mod *-x *wrap.cxx

#===============================================================================
# Rules
#===============================================================================

.SUFFIXES: .F90 .o
.PHONY: clean neat

%.o: %.f90
	$(F90) $(F90FLAGS) $(INCLUDE) -c $<
