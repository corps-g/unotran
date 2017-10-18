# Makefile for linear algebra tests 

#===============================================================================
# Compilation settings
#===============================================================================

F90 = gfortran
F90FLAGS = -O0 -ffree-line-length-none -fcheck=all -fdefault-double-8 -fdefault-real-8
LDFLAGS = 


#===============================================================================
# Object Files
#===============================================================================

objects =        \
angle.o          \
control.o        \
dgm.o            \
dgmsolver.o      \
dgmsweeper.o     \
material.o       \
mesh.o           \
solver.o         \
state.o          \
sweeper.o
 
#===============================================================================
# Targets
#===============================================================================

# main executable
xdgm: $(objects) main.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o $@ 

# executable tests
test_angle: $(objects) ./test/test_angle.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_dgm: $(objects) ./test/test_dgm.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_dgmsolver: $(objects) ./test/test_dgmsolver.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_dgmsweeper: $(objects) ./test/test_dgmsweeper.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_material: $(objects) ./test/test_material.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_mesh: $(objects) ./test/test_mesh.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_solver: $(objects) ./test/test_solver.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_state: $(objects) ./test/test_state.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test_sweeper: $(objects) ./test/test_sweeper.f90
	$(F90) $(F90FLAGS) $(LDFLAGS) $^ -o ./test/$@.x
	./test/$@.x
test: test_angle test_mesh test_material test_state test_sweeper test_solver test_dgm test_dgmsweeper test_dgmsolver

# f2py module
pydgm: control.f90 angle.f90 material.f90 mesh.f90 #state.f90 sweeper.f90 solver.f90 dgm.f90 dgmsweeper.f90 dgmsolver.f90
	f2py --f90flags="$(F90FLAGS)" -c -m $@ $^

# Remove objects, module files, and exectables
clean:
	@rm -f *.o *.mod test/*.x


#===============================================================================
# Rules
#===============================================================================

.SUFFIXES: .F90 .o
.PHONY: clean neat

%.o: %.f90
	$(F90) $(F90FLAGS) $(INCLUDE) -c $<


#===============================================================================
# Dependencies
#===============================================================================

angle.o: control.o
dgm.o: control.o
dgm.o: material.o
dgm.o: mesh.o
dgm.o: angle.o
dgm.o: state.o
dgmsolver.o: control.o
dgmsolver.o: material.o
dgmsolver.o: angle.o
dgmsolver.o: mesh.o
dgmsolver.o: state.o
dgmsolver.o: dgm.o
dgmsolver.o: dgmsweeper.o
dgmsweeper.o: control.o
dgmsweeper.o: material.o
dgmsweeper.o: mesh.o
dgmsweeper.o: angle.o
dgmsweeper.o: sweeper.o
dgmsweeper.o: state.o
dgmsweeper.o: dgm.o
main.o: control.o
main.o: dgmsolver.o
main.o: solver.o
material.o: control.o
mesh.o: control.o
solver.o: control.o
solver.o: material.o
solver.o: angle.o
solver.o: mesh.o
solver.o: state.o
solver.o: sweeper.o
state.o: control.o
state.o: material.o
state.o: mesh.o
state.o: angle.o
sweeper.o: control.o
sweeper.o: material.o
sweeper.o: mesh.o
sweeper.o: angle.o
sweeper.o: state.o
