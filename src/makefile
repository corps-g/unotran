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

vpath test_%.f90 
BC := /usr/bin/bc

objects = material mesh angle sweeper state

all: $(objects) main

# Use the repository's BLAS implementation
angle: angle.f90
	f2py2.7 --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@ 
	
test_angle: angle.f90 ./test/test_angle.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_angle.so $^ 
	@./test/test_angle.so
	
mesh: mesh.f90
	f2py2.7 --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@ 
	
test_mesh: mesh.f90 ./test/test_mesh.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_mesh.so $^ 
	@./test/test_mesh.so
	
material: material.f90
	f2py2.7 --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_material: material.f90 ./test/test_material.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_material.so $^ 
	@./test/test_material.so
	
state: material.f90 mesh.f90 angle.f90 state.f90
	f2py2.7 --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_state: mesh.f90 material.f90 angle.f90 state.f90 ./test/test_state.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_state.so $^ 
	@./test/test_state.so
	
sweeper: material.f90 mesh.f90 angle.f90 state.f90 sweeper.f90
	f2py2.7 --fcompiler=$(F90) $(LIBBLAS) -c $^ -m $@
	
test_sweeper: mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 ./test/test_sweeper.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_sweeper.so $^ 
	@./test/test_sweeper.so

main: material.f90 mesh.f90 angle.f90 state.f90 sweeper.f90 main.f90
	$(F90) $(OMPFLAG) $(LIBBLAS) -o main.exe $^ 
	
test_main: mesh.f90 material.f90 angle.f90 state.f90 sweeper.f90 ./test/test_main.f90
	@$(F90) $(OMPFLAG) $(LIBBLAS) -o ./test/test_main.so $^ 
	@./test/test_main.so

test: test_angle test_mesh test_material test_state test_sweeper test_main

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
