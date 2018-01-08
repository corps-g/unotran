import sys
sys.path.append('../')

import unittest
import pydgm
import numpy as np

class TestDGMSOLVER(unittest.TestCase):

    def setUp(self):
        # Set the variables for the test
        pydgm.control.fine_mesh = [1]
        pydgm.control.coarse_mesh = [0.0, 1.0]
        pydgm.control.material_map = [1]
        pydgm.control.xs_name = 'test/2gXS.anlxs'.ljust(256)
        pydgm.control.angle_order = 2
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [0.0, 0.0]
        pydgm.control.allow_fission = False
        pydgm.control.energy_group_map = [1]
        pydgm.control.dgm_basis_name = '2gbasis'.ljust(256)
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 1.0
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.ignore_warnings = True

    def test_dgmsolver_2gtest(self):
        '''
        Test the 2g dgm fixed source problem
        '''
        phi_test = np.reshape([1.1420149990909008, 0.37464706668551212], (1, 2, 1), 'F')
        psi_test = np.reshape([0.81304488744042813, 0.29884810509581583, 1.31748796916478740, 0.41507830480599001, 1.31748796916478740, 0.41507830480599001, 0.81304488744042813, 0.29884810509581583], (2, 4, 1))

        pydgm.dgmsolver.initialize_dgmsolver()

        pydgm.state.phi = phi_test
        pydgm.state.psi = psi_test

        phi_new = np.reshape(np.zeros(2), phi_test.shape, 'F')
        psi_new = np.reshape(np.zeros(8), psi_test.shape, 'F')

        pydgm.dgmsweeper.dgmsweep(phi_new, psi_new, pydgm.dgmsolver.incoming)

        np.testing.assert_array_almost_equal(pydgm.state.phi, phi_test, 12)
        np.testing.assert_array_almost_equal(pydgm.state.psi, psi_test, 12)

    def test_dgmsolver_vacuum1(self):
        '''
        Test the 7g dgm fixed source problem with vacuum boundary conditions
        '''
        # Set the variables for the test
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [6, 1, 6]
        pydgm.control.xs_name = 'test.anlxs'.ljust(256)
        pydgm.control.angle_order = 10
        pydgm.control.allow_fission = True
        pydgm.control.energy_group_map = [4]
        pydgm.control.dgm_basis_name = 'basis'.ljust(256)
        pydgm.control.lamb = 0.75
        
        pydgm.dgmsolver.initialize_dgmsolver()

        phi_test = [1.8461980363287278, 4.16904423296633, 1.6371065827443676, 1.3260798327554102, 1.1507280372172657, 1.221454815786205, 0.21787346591975096, 1.8779676212462644, 4.346293616739397, 1.7911796347394968, 1.4696141054340566, 1.2762011622747407, 1.410548033592605, 0.2805431110268466, 1.9076720822947586, 4.5235324366607985, 1.9348875590099663, 1.5995201255730067, 1.3850154736028073, 1.5361742894430894, 0.3503620127705863, 1.9653381411150332, 4.755704786297735, 2.0602776849402016, 1.7086687870124164, 1.473314427190565, 1.6032397558933036, 0.41859495632314897, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.794379906316901, 1.5415868503767243, 1.6402668270670566, 0.45783584286439, 2.1183287442900616, 5.259894345939496, 2.249750972825359, 1.8662129456617995, 1.598567531294177, 1.6740654516371898, 0.48316274993701736, 2.180431853934209, 5.463189536952539, 2.323379278194582, 1.9263940668089787, 1.6461554034562276, 1.7042440374045473, 0.5017303824642151, 2.233671544963958, 5.636838362725401, 2.3851794675942126, 1.976556103614348, 1.6857277137465003, 1.7306376835631267, 0.5160940364316613, 2.278501420880871, 5.782620523285358, 2.43635508075605, 2.0178957585753343, 1.7182838252262809, 1.753193622571626, 0.5273657105739336, 2.3153010315366376, 5.902002999445916, 2.4778225682576234, 2.051286995516379, 1.7445473309392545, 1.7719149818882431, 0.5361455617094784, 2.344379734887513, 5.996163165441185, 2.5102702850685095, 2.0773624527330834, 1.7650382804142186, 1.7868302159372544, 0.5428111210518882, 2.365980375010593, 6.066012532308792, 2.534202681742982, 2.096571354470798, 1.7801238463024747, 1.7979755323610467, 0.5476165734363021, 2.380282353001199, 6.112216861028091, 2.54997251963054, 2.1092199413577197, 1.790053258345466, 1.8053845815737255, 0.5507325908285529, 2.3874039642095606, 6.135211472153911, 2.557803276430926, 2.115498615897139, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.3874039642095597, 6.1352114721539115, 2.5578032764309264, 2.1154986158971383, 1.7949810125894576, 1.809082468016729, 0.5522651544630954, 2.380282353001199, 6.112216861028088, 2.5499725196305403, 2.1092199413577197, 1.790053258345467, 1.8053845815737257, 0.5507325908285529, 2.365980375010593, 6.0660125323087914, 2.534202681742981, 2.096571354470798, 1.7801238463024753, 1.797975532361047, 0.547616573436302, 2.3443797348875135, 5.996163165441185, 2.5102702850685086, 2.0773624527330825, 1.7650382804142186, 1.7868302159372544, 0.5428111210518881, 2.315301031536638, 5.9020029994459176, 2.477822568257624, 2.0512869955163793, 1.7445473309392545, 1.7719149818882434, 0.5361455617094785, 2.278501420880872, 5.782620523285358, 2.436355080756051, 2.0178957585753348, 1.718283825226282, 1.753193622571626, 0.5273657105739337, 2.233671544963958, 5.636838362725398, 2.3851794675942113, 1.976556103614348, 1.6857277137465008, 1.7306376835631265, 0.516094036431661, 2.1804318539342096, 5.463189536952541, 2.323379278194582, 1.9263940668089785, 1.646155403456228, 1.704244037404548, 0.5017303824642152, 2.1183287442900616, 5.259894345939497, 2.2497509728253595, 1.8662129456617995, 1.5985675312941772, 1.6740654516371902, 0.48316274993701747, 2.046831950358654, 5.024852817609021, 2.1627268750309767, 1.7943799063169015, 1.5415868503767245, 1.640266827067057, 0.45783584286439005, 1.9653381411150337, 4.755704786297734, 2.060277684940202, 1.708668787012417, 1.4733144271905656, 1.6032397558933043, 0.4185949563231489, 1.9076720822947586, 4.5235324366608, 1.9348875590099668, 1.5995201255730072, 1.385015473602808, 1.53617428944309, 0.35036201277058626, 1.8779676212462644, 4.346293616739396, 1.791179634739497, 1.469614105434057, 1.2762011622747407, 1.4105480335926057, 0.2805431110268464, 1.8461980363287276, 4.169044232966329, 1.6371065827443676, 1.3260798327554109, 1.1507280372172666, 1.2214548157862055, 0.21787346591975099]
        pydgm.state.phi[0,:,:] = np.reshape(phi_test, (7, 28), 'F')

        pydgm.dgmsolver.dgmsolve()

        np.testing.assert_array_almost_equal(pydgm.state.phi[0].flatten('F'), phi_test, 12)

    def test_dgmsolver_reflect1(self):
        pydgm.control.fine_mesh = [3, 22, 3]
        pydgm.control.coarse_mesh = [0.0, 0.09, 1.17, 1.26]
        pydgm.control.material_map = [1, 1, 1]
        pydgm.control.xs_name = 'test.anlxs'.ljust(256)
        pydgm.control.angle_order = 10
        pydgm.control.angle_option = pydgm.angle.gl
        pydgm.control.boundary_type = [1.0, 1.0]
        pydgm.control.allow_fission = False
        pydgm.control.energy_group_map = [4]
        pydgm.control.dgm_basis_name = 'basis'.ljust(256)
        pydgm.control.outer_print = False
        pydgm.control.inner_print = False
        pydgm.control.outer_tolerance = 1e-14
        pydgm.control.inner_tolerance = 1e-14
        pydgm.control.lamb = 0.3
        pydgm.control.use_dgm = True
        pydgm.control.store_psi = True
        pydgm.control.solver_type = 'fixed'.ljust(256)
        pydgm.control.source_value = 1.0
        pydgm.control.equation_type = 'DD'
        pydgm.control.legendre_order = 0
        pydgm.control.ignore_warnings = True
        pydgm.control.max_inner_iters = 10
        pydgm.control.max_outer_iters = 5000
        
        pydgm.dgmsolver.initialize_dgmsolver()
        
        # Compute the test flux
        T = np.diag(pydgm.material.sig_t[:,0])
        S = pydgm.material.sig_s[0,:,:,0].T
        phi_test = np.linalg.solve((T - S), np.ones(7))
        phi_test = np.array([phi_test for i in range(28)]).flatten()
        
        pydgm.dgmsolver.dgmsolve()
        
        np.testing.assert_array_almost_equal(pydgm.state.phi[0, :, :].flatten('F') / phi_test, np.ones(28 * 7), 12)

    def test_dgmsolver_vacuum2(self):
        '''
        use control
        use dgmsolver
        
        implicit none
        
        ! initialize types
        integer :: fineMesh(1), materialMap(1), l, c, a, g, counter, testCond, t1, t2
        double precision :: coarseMesh(2), norm, error, phi_test(7,1), &
                            boundary(2), psi_test(7,4,1)
        
        call initialize_control('test/dgm_test_options3', .true.)
        call initialize_dgmsolver()
        
        phi_test = reshape([1.1076512516190389, 1.1095892550819531,&
                            1.0914913168898499, 1.0358809957845283,&
                            0.93405352272848619, 0.79552760081182894,&
                            0.48995843862242699],shape(phi_test))
        
        psi_test = reshape([ 0.62989551954274092,0.65696484059125337,&
                             0.68041606804080934,0.66450867626366705,&
                             0.60263096140806338,0.53438683380855967,&
                             0.38300537939872042,1.3624866129734592,&
                             1.3510195477616225,1.3107592451448140,&
                             1.2339713438183100,1.1108346317861364,&
                             0.93482033401344933,0.54700730201708170,&
                             1.3624866129734592,1.3510195477616225,&
                             1.3107592451448140,1.2339713438183100,&
                             1.1108346317861364,0.93482033401344933,&
                             0.54700730201708170,0.62989551954274092,&
                             0.65696484059125337,0.68041606804080934,&
                             0.66450867626366705,0.60263096140806338,&
                             0.53438683380855967,0.38300537939872042], &
                             shape(psi_test))
        
        phi(0,:,:) = phi_test
        psi(:,:,:) = psi_test
        source(:,:,:) = 1.0
        
        call dgmsolve()
        
        t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)
        
        if (t1 == 0) then
          print *, 'dgmsolver: vacuum test 2 failed'
        else
          print *, 'all tests passed for dgmsolver vacuum 2'
        end if
        
        call finalize_dgmsolver()
        call finalize_control()
        '''

    def test_dgmsolver_reflect2(self):
        '''
        use control
        use dgmsolver
        
        implicit none
        
        ! initialize types
        integer :: fineMesh(1), materialMap(1), l, c, a, g, counter, testCond, t1, t2
        double precision :: coarseMesh(2), norm, error, phi_test(7,1), &
                            psi_test(7,4,1), boundary(2)
        
        call initialize_control('test/dgm_test_options3', .true.)
        boundary_type = [1.0, 1.0]
        call initialize_dgmsolver()
        
        source = 1.0
        
        phi_test = reshape([94.512658438949273,106.66371642824166,&
                            75.397102078564259,17.953651480951333,&
                            6.2855008963667123,3.0158479735464110,&
                            1.2132770548341645],shape(phi_test))
        
        psi_test = reshape([94.512658438949273,106.66371642824166,&
                            75.397102078564259,17.953651480951333,&
                            6.2855008963667123,3.0158479735464110,&
                            1.2132770548341645,94.512658454844384,&
                            106.66371644092482,75.397102082192546,&
                            17.953651480951343,6.2855008963667141,&
                            3.0158479735464105,1.2132770548341649,&
                            94.512658454844384,106.66371644092482,&
                            75.397102082192546,17.953651480951343,&
                            6.2855008963667141,3.0158479735464105,&
                            1.2132770548341649,94.512658438949273,&
                            106.66371642824166,75.397102078564259,&
                            17.953651480951333,6.2855008963667123,&
                            3.0158479735464110,1.2132770548341645], shape(psi_test))
        
        phi(0,:,:) = phi_test
        psi(:,:,:) = psi_test
        source(:,:,:) = 1.0
        
        call dgmsolve()
        
        t1 = testCond(all(abs(phi(0,:,:) - phi_test) < 1e-12))
        
        if (t1 == 0) then
          print *, 'dgmsolver: reflection test 2 failed'
        else
          print *, 'all tests passed for dgmsolver reflect 2'
        end if
        
        call finalize_dgmsolver()
        call finalize_control()
        '''
        
    def test_dgmsolver_eigenV2g(self):
        '''
        use control
        use dgmsolver
        use angle, only : wt
        
        implicit none
        
        ! initialize types
        integer :: testCond, t1, t2, t3, a, c
        double precision :: phi_test(2,10), psi_test(2,4,10), keff_test
        
        ! Define problem parameters
        call initialize_control('test/eigen_test_options', .true.)
        xs_name = 'test/2gXS.anlxs'
        source_value = 0.0
        boundary_type = [0.0, 0.0]
        legendre_order = 0
        dgm_basis_name = '2gbasis'
        use_DGM = .true.
        use_recondensation = .false.
        outer_print = .false.
        inner_print = .false.
        lambda = 0.25
        max_inner_iters = 50
        max_outer_iters = 5000
        ignore_warnings = .true.
        
        !  call output_control()
        
        call initialize_dgmsolver()
        
        keff_test = 0.8099523232983425
        
        phi_test = reshape([&
                    0.05882749189352335, 0.009858087422743867, 0.1099501419733834, &
                    0.01934781232927005, 0.1497992040255296, &
                    0.02617665800594787, 0.1781310366743199, 0.03122421632573503, &
                    0.1928663608358422, 0.03377131381204261, &
                    0.1928663608358422, 0.03377131381204262, 0.1781310366743200, &
                    0.03122421632573504, 0.1497992040255296, &
                    0.02617665800594788, 0.1099501419733835, 0.01934781232927007, &
                    0.05882749189352339, 0.009858087422743878],shape(phi_test))
        
        call dgmsolve()
        
        phi = phi / phi(0,1,1) * phi_test(1,1)
        
        t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)
        
        phi_test = 0
        do c = 1, number_cells
          do a = 1, number_angles
            phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
            phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                                * psi(:, 2 * number_angles - a + 1,c)
          end do
        end do
        phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
        t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 20) < 1e-12))
        
        t3 = testCond(abs(d_keff - keff_test) < 1e-12)
        
        if (t1 == 0) then
          print *, 'dgmsolver: eigen 2g vacuum solver phi failed'
        else if (t2 == 0) then
          print *, 'dgmsolver: eigen 2g vacuum solver psi failed'
        else if (t3 == 0) then
          print *, 'dgmsolver: eigen 2g vacuum solver keff failed'
        else
          print *, 'all tests passed for eigen 2g vacuum dgmsolver'
        end if
        
        call finalize_dgmsolver()
        call finalize_control()
        '''
        
    def test_dgmsolver_eigenV7g(self):
        '''
        use control
        use dgmsolver
        use angle, only : wt
        
        implicit none
        
        ! initialize types
        integer :: testCond, t1, t2, t3, a, c
        double precision :: phi_test(7,10), psi_test(7,4,10), keff_test
        
        ! Define problem parameters
        call initialize_control('test/eigen_test_options', .true.)
        xs_name = 'test/testXS.anlxs'
        source_value = 0.0
        boundary_type = [0.0, 0.0]
        legendre_order = 0
        dgm_basis_name = 'basis'
        use_DGM = .true.
        use_recondensation = .false.
        energy_group_map = [4]
        outer_print = .true.
        inner_print = .false.
        lambda = 0.4
        max_inner_iters = 10
        max_outer_iters = 5000
        ignore_warnings = .true.
        
        call initialize_dgmsolver()
        
        keff_test = 0.4347999090254699
        
        phi_test = reshape([&
                    0.4054826380458894, 0.7797540846047309, 0.03343005518884316, &
                    3.651031406190026e-05, 0.0000000000000000, &
                    0.0000000000000000, 0.0000000000000000, 0.5443708124629247, &
                    1.1117877528804340, 0.05133797579519819, &
                    5.786890101941508e-05, 0.0000000000000000, 0.0000000000000000, &
                    0.0000000000000000, 0.6541468489653601, &
                    1.3652812748897960, 0.06471565393213054, 7.336542490037297e-05, &
                    0.0000000000000000, 0.0000000000000000, &
                    0.0000000000000000, 0.7300062250116002, 1.5374126354668270, &
                    0.07375213451710246, 8.374721450748013e-05, &
                    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, &
                    0.7687444954671941, 1.6246002203441580, &
                    0.07832746418803245, 8.899077210821326e-05, 0.0000000000000000, &
                    0.0000000000000000, 0.0000000000000000, &
                    0.7687444954671943, 1.6246002203441580, 0.07832746418803245, &
                    8.899077210821326e-05, 0.0000000000000000, &
                    0.0000000000000000, 0.0000000000000000, 0.7300062250116002, &
                    1.5374126354668280, 0.07375213451710246, &
                    8.374721450748013e-05, 0.0000000000000000, 0.0000000000000000, &
                    0.0000000000000000, 0.6541468489653601, &
                    1.3652812748897960, 0.06471565393213052, 7.336542490037296e-05, &
                    0.0000000000000000, 0.0000000000000000, &
                    0.0000000000000000, 0.5443708124629247, 1.1117877528804350, &
                    0.05133797579519818, 5.786890101941509e-05, &
                    0.0000000000000000, 0.0000000000000000, 0.0000000000000000, &
                    0.4054826380458894, 0.7797540846047309, &
                    0.03343005518884316, 3.651031406190026e-05, 0.0000000000000000, &
                    0.0000000000000000, 0.0000000000000000],shape(phi_test))
        
        call dgmsolve()
        
        print *, phi(0,:,:)
        
        phi_test = phi_test / phi_test(1, 1) * phi(0, 1, 1)
        
        print *
        print *, phi_test
        
        t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)
        
        phi_test = 0
        do c = 1, number_cells
          do a = 1, number_angles
            phi_test(:, c) = phi_test(:, c) + 0.5 * wt(a) * psi(:, a, c)
            phi_test(:, c) = phi_test(:, c) + 0.5 * wt(number_angles - a + 1) &
                                                * psi(:, 2 * number_angles - a + 1, c)
          end do
        end do
        t2 = testCond(all(abs(phi_test(:,:) - phi(0,:,:)) < 1e-12))
        
        t3 = testCond(abs(d_keff - keff_test) < 1e-12)
        
        if (t1 == 0) then
          print *, 'solver: eigen 7g vacuum solver phi failed'
        else if (t2 == 0) then
          print *, 'solver: eigen 7g vacuum solver psi failed'
        else if (t3 == 0) then
          print *, 'solver: eigen 7g vacuum solver keff failed'
        else
          print *, 'all tests passed for eigen 7g vacuum solver'
        end if
        
        call finalize_dgmsolver()
        call finalize_control()
        '''
        
    def test_dgmsolver_eigenR2g(self):
        '''
        use control
        use dgmsolver
        
        implicit none
        
        ! initialize types
        integer :: testCond, t1, t2, t3, g
        double precision :: phi_test(2), keff_test, psi_test(2)
        
        ! Define problem parameters
        call initialize_control('test/eigen_test_options', .true.)
        xs_name = 'test/2gXS.anlxs'
        legendre_order = 0
        dgm_basis_name = '2gbasis'
        use_DGM = .true.
        use_recondensation = .false.
        outer_print = .false.
        inner_print = .false.
        lambda = 0.1
        max_inner_iters = 10
        max_outer_iters = 5000
        ignore_warnings = .true.
        
        call initialize_dgmsolver()
        
        call dgmsolve()
        
        keff_test = 0.8403361344537817
        
        phi_test = [0.1428571428571399, 0.0252100840336125]
        psi_test = [0.07142857142856951, 0.0126050420168061]
        
        phi = phi / phi(0,1,1) * phi_test(1)
        psi = psi / psi(1,1,1) * psi_test(1)
        
        do g = 1, 2
          phi(0,g,:) = phi(0,g,:) - phi_test(g)
          psi(g,:,:) = psi(g,:,:) - psi_test(g)
        end do
        
        t1 = testCond(all(abs(phi) < 1e-6))
        t2 = testCond(all(abs(psi) < 1e-6))
        t3 = testCond(abs(d_keff - keff_test) < 1e-12)
        
        if (t1 == 0) then
          print *, 'solver: eigen 2g reflection solver phi failed'
        else if (t2 == 0) then
          print *, 'solver: eigen 2g reflection solver psi failed'
        else if (t3 == 0) then
          print *, 'solver: eigen 2g reflection solver keff failed'
        else
          print *, 'all tests passed for eigen 2g reflection solver'
        end if
        
        call finalize_dgmsolver()
        call finalize_control
        '''
        
    def test_dgmsolver_eigenR7g(self):
        '''
        use control
        use dgmsolver
        use dgm, only : energyMesh, basis
        
        implicit none
        
        ! initialize types
        integer :: testCond, t1, t2, t3, g, c, a, cg, i
        double precision :: phi_test(7), keff_test, psi_test(7)
        
        ! Define problem parameters
        call initialize_control('test/eigen_test_options', .true.)
        xs_name = 'test/testXS.anlxs'
        legendre_order = 0
        dgm_basis_name = 'basis'
        use_DGM = .true.
        use_recondensation = .false.
        energy_group_map = [4]
        outer_print = .false.
        inner_print = .false.
        lambda = 0.1
        max_inner_iters = 1000
        max_outer_iters = 5000
        ignore_warnings = .true.
        
        !call output_control()
        
        keff_test = 2.4249465383700772
        
        phi_test = [3.9968764019364260, 5.3155446123379970, 0.2410710372839961, &
                    0.0001149482887246657, 0.0000000000000000, &
                    0.0000000000000000, 0.0000000000000000]
        psi_test = [1.9984382009682120, 2.6577723061689960, 0.1205355186419960, &
                    5.747414436169997e-05, 0.0000000000000000, &
                    0.0000000000000000, 0.0000000000000000]
        
        call initialize_dgmsolver()
        
        do c=1, number_cells
          do a=1, number_angles * 2
            if (a == 1) phi(0,:,c) = phi_test
            psi(:,a,c) = phi_test
          end do
        end do
        
        incoming = 0
        do i = 0, expansion_order
          do a = 1, number_angles
            do g = 1, number_groups
              cg = energyMesh(g)
              ! Angular flux
              incoming(cg, a, i) = incoming(cg, a, i) +  basis(g, i) * psi(g, a, 1)
            end do
          end do
        end do
        
        d_keff = keff_test
        
        call dgmsolve()
        
        phi = phi / phi(0,1,1) * phi_test(1)
        psi = psi / psi(1,1,1) * psi_test(1)
        
        do g = 1, 7
          phi(0,g,:) = phi(0,g,:) - phi_test(g)
          psi(g,:,:) = psi(g,:,:) - psi_test(g)
        end do
        
        t1 = testCond(all(abs(phi(0,:,:)) < 1e-12))
        t2 = testCond(all(abs(psi) < 1e-12))
        t3 = testCond(abs(d_keff - keff_test) < 1e-12)
        
        if (t1 == 0) then
          print *, 'solver: eigen 7g reflection solver phi failed'
        else if (t2 == 0) then
          print *, 'solver: eigen 7g reflection solver psi failed'
        else if (t3 == 0) then
          print *, 'solver: eigen 7g reflection solver keff failed'
        else
          print *, 'all tests passed for eigen 7g reflection solver'
        end if
        
        call finalize_dgmsolver()
        call finalize_control
        '''

    def test_dgmsolver_eigenR2gPin(self):
        '''
        use control
        use dgmsolver
        use angle, only : wt
        
        implicit none
        
        ! initialize types
        integer :: testCond, t1, t2, t3, a, c
        double precision :: phi_test(2,16), psi_test(2,4,16), keff_test
        
        ! Define problem parameters
        call initialize_control('test/pin_test_options', .true.)
        xs_name = 'test/2gXS.anlxs'
        
        call initialize_dgmsolver()
        
        keff_test = 0.8418546852484950
        
        phi_test = reshape([0.133931831084, 0.0466324063141, 0.134075529413, &
                            0.0455080808626, 0.1343633343, &
                            0.0432068414739, 0.13516513934, 0.0384434752118, &
                            0.136157377422, 0.0332992956, &
                            0.136742846609, 0.0304645081033, 0.137069783633, &
                            0.0289701995061, 0.1372163852, &
                            0.0283256746626, 0.137216385156, 0.0283256746626, &
                            0.137069783633, 0.02897019951, &
                            0.136742846609, 0.0304645081033, 0.136157377422, &
                            0.0332992956043, 0.1351651393, &
                            0.0384434752119, 0.134363334286, 0.0432068414741, &
                            0.134075529413, 0.04550808086, &
                            0.133931831085, 0.0466324063142],shape(phi_test))
        
        call dgmsolve()
        
        phi_test = phi_test / phi_test(1,1) * phi(0,1,1)
        psi_test = psi_test / psi_test(1,1,1) * psi(1,1,1)
        
        t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)
        
        phi_test = 0
        do c = 1, number_cells
          do a = 1, number_angles
            phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
            phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                                * psi(:, 2 * number_angles - a + 1,c)
          end do
        end do
        phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
        t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 32) < 1e-12))
        
        t3 = testCond(abs(d_keff - keff_test) < 1e-12)
        
        if (t1 == 0) then
          print *, 'solver: eigen 2g reflect pin solver phi failed'
        else if (t2 == 0) then
          print *, 'solver: eigen 2g reflect pin solver psi failed'
        else if (t3 == 0) then
          print *, 'solver: eigen 2g reflect pin solver keff failed'
        else
          print *, 'all tests passed for eigen 2g reflect pin solver'
        end if
        
        call finalize_dgmsolver()
        call finalize_control()
        '''
    
    def test_dgmsolver_eigenR7gPin(self):
        '''
        use control
        use dgmsolver
        use angle, only : wt
        
        implicit none
        
        ! initialize types
        integer :: testCond, t1, t2, t3, c, a
        double precision :: phi_test(7,16), psi_test(7,4,16), keff_test
        
        ! Define problem parameters
        call initialize_control('test/pin_test_options', .true.)
        xs_name = 'test/testXS.anlxs'
        material_map = [6, 1, 6]
        
        call initialize_dgmsolver()
        
        keff_test = 1.3515451561148644
        
        phi_test = reshape([2.2650616717, 2.8369334989, 0.0787783060158, &
                            7.04175236386e-05, 3.247808073e-11, &
                            2.42090423431e-13, 1.5855325733e-13, 2.26504648629, &
                            2.83730954426, 0.07884130784, &
                            7.05024889238e-05, 3.23070059067e-11, 2.416262144e-13, &
                            1.50689303232e-13, 2.2650161, &
                            2.83806173523, 0.0789674406093, 7.06726888056e-05, &
                            3.19637591127e-11, 2.406809989e-13, &
                            1.33594887863e-13, 2.26484830727, 2.84044264348, &
                            0.0792086051569, 7.098912578e-05, &
                            3.13235926512e-11, 2.39206404496e-13, 1.09840721921e-13, &
                            2.26458581485, 2.843834874, &
                            0.0794963139764, 7.13605988661e-05, 3.05732041796e-11, &
                            2.37613706246e-13, 8.964680919e-14, &
                            2.26439199657, 2.84637204739, 0.0797108850311, &
                            7.16364816476e-05, 3.001907576e-11, &
                            2.36383833154e-13, 7.77296907704e-14, 2.26426423355, &
                            2.84806017199, 0.07985335909, &
                            7.18191179175e-05, 2.96537913389e-11, 2.35550014439e-13, &
                            7.10923697151e-14, 2.264200786, &
                            2.8489032388, 0.0799244247599, 7.1910052492e-05, &
                            2.94724771287e-11, 2.351334627e-13, &
                            6.81268597471e-14, 2.26420078604, 2.8489032388, &
                            0.0799244247599, 7.191005249e-05, &
                            2.9472717738e-11, 2.3514457477e-13, 6.81336094823e-14, &
                            2.26426423355, 2.848060172, &
                            0.0798533590933, 7.18191179264e-05, 2.96545146531e-11, &
                            2.35583516359e-13, 7.111336685e-14, &
                            2.26439199657, 2.84637204739, 0.0797108850311, &
                            7.16364816625e-05, 3.002028626e-11, &
                            2.36440226704e-13, 7.77673697266e-14, 2.26458581485, &
                            2.84383487412, 0.07949631398, &
                            7.1360598887e-05, 3.05749093972e-11, 2.37693842494e-13, &
                            8.97059192338e-14, 2.264848307, &
                            2.84044264348, 0.0792086051571, 7.09891258041e-05, &
                            3.13258032808e-11, 2.393115064e-13, &
                            1.09929579303e-13, 2.26501610046, 2.83806173523, &
                            0.0789674406095, 7.067268884e-05, &
                            3.19663665349e-11, 2.40808670657e-13, 1.33719616187e-13, &
                            2.26504648629, 2.837309544, &
                            0.0788413078367, 7.05024889589e-05, 3.23098965985e-11, &
                            2.41773215344e-13, 1.508514105e-13, &
                            2.2650616717, 2.8369334989, 0.078778306016, &
                            7.04175236773e-05, 3.248125734e-11, &
                            2.42256465053e-13, 1.58755730155e-13],shape(phi_test))
        
        call dgmsolve()
        
        phi_test = phi_test / phi_test(1,1) * phi(0,1,1)
        
        t1 = testCond(norm2(phi(0,:,:) - phi_test) < 1e-12)
        
        phi_test = 0
        do c = 1, number_cells
          do a = 1, number_angles
            phi_test(:,c) = phi_test(:,c) + 0.5 * wt(a) * psi(:,a,c)
            phi_test(:,c) = phi_test(:,c) + 0.5 * wt(number_angles - a + 1) &
                                                * psi(:, 2 * number_angles - a + 1,c)
          end do
        end do
        phi_test(:,:) = phi_test(:,:) / phi(0,:,:)
        t2 = testCond(all(abs(phi_test(:,:) - sum(phi_test) / 112) < 1e-12))
        
        t3 = testCond(abs(d_keff - keff_test) < 1e-12)
        
        if (t1 == 0) then
          print *, 'solver: eigen 7g reflect pin solver phi failed'
        else if (t2 == 0) then
          print *, 'solver: eigen 7g reflect pin solver psi failed'
        else if (t3 == 0) then
          print *, 'solver: eigen 7g reflect pin solver keff failed'
        else
          print *, 'all tests passed for eigen 7g reflect pin solver'
        end if
        
        call finalize_dgmsolver()
        call finalize_control()
        '''
    
    def tearDown(self):
        pydgm.dgmsolver.finalize_dgmsolver()
        pydgm.control.finalize_control()

if __name__ == '__main__':

    unittest.main()

