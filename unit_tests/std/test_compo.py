import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import scipy.sparse as sp
import dustpy.constants as c
from tripod import Simulation

from tripod.std.compo import (
    prepare, set_state_vector_components, finalize, Y_jacobian,
    _f_impl_1_direct_compo, jacobian_compo, A_grains, L_condensation,
    L_sublimation, c_jacobian, set_boundaries_component
)

class TestStateVectorManagement:

    @pytest.fixture
    def sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        return sim
    
    @pytest.fixture
    def Sigma_gas(self):
        return np.array([100.0, 200.0, 300.0])
    
    @pytest.fixture
    def Sigma_dust(self):
        return np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    

    def test_set_state_vector_components_gas_active(self,sim,Sigma_gas):
        """Test state vector setup for gas active component"""

        sim.addcomponent_c(name="water", gas_value=Sigma_gas, mu=18.0, gas_active=True)
        comp_gas = sim.components.__dict__["water"]
        set_state_vector_components(sim)
        
        # Verify state vector was set correctly
        np.testing.assert_array_equal(comp_gas._Y, comp_gas.gas.Sigma)
        np.testing.assert_array_equal(comp_gas._S, comp_gas.gas.Sigma_dot)
    
    def test_set_state_vector_components_dust_tracer(self,sim,Sigma_dust):
        """Test state vector setup for dust tracer component"""

        sim.addcomponent_c(name="silicate_tracer", gas_value=0.0, mu=0.0, dust_value=Sigma_dust, dust_tracer=True)
        comp_dust_tracer = sim.components.__dict__["silicate_tracer"]
        set_state_vector_components(sim)
        
        # Verify state vector calculation
        expected_Y = (comp_dust_tracer.dust.value * sim.dust.Sigma).ravel()
        np.testing.assert_array_equal(comp_dust_tracer._Y, expected_Y)
    
    def test_set_state_vector_components_dust_active(self,sim,Sigma_dust):
        """Test state vector setup for active dust component"""
        
        sim.addcomponent_c(name="carbon_dust", gas_value=0.0, mu=0.0, dust_value=Sigma_dust, dust_active=True)
        comp_dust = sim.components.__dict__["carbon_dust"]
        set_state_vector_components(sim)
        
        # Verify state vector was set correctly
        expected_Y = comp_dust.dust.Sigma.ravel()
        np.testing.assert_array_equal(comp_dust._Y, expected_Y)
        np.testing.assert_array_equal(comp_dust._S, comp_dust.dust.S.ext.ravel())
    
    def test_set_state_vector_components_mixed_dust_gas(self,sim,Sigma_gas,Sigma_dust):
        """Test state vector setup for mixed dust and gas component"""

        sim.addcomponent_c(name="mixed_compo", gas_value=Sigma_gas, mu=18.0, dust_value=Sigma_dust, dust_active=True, gas_active=True)
        comp_mixed = sim.components.__dict__["mixed_compo"]

        set_state_vector_components(sim)
        
        Nr = sim.grid.Nr
        # Verify gas part
        np.testing.assert_array_equal(comp_mixed._Y[:Nr], comp_mixed.gas.Sigma)
        np.testing.assert_array_equal(comp_mixed._S[:Nr], comp_mixed.gas.Sigma_dot)
        
        # Verify dust part
        expected_dust_Y = (comp_mixed.dust.Sigma).ravel()
        np.testing.assert_array_equal(comp_mixed._Y[Nr:], expected_dust_Y)

    
class TestFinalization:
    @pytest.fixture
    def sim(self):
        sim = Simulation()
        sim.ini.grid.Nr = 3
        sim.initialize()
        sim.t.snapshots = [1.0]
        sim.writer = None
        return sim


    def test_finalize_updates_tracer(self, sim):
        """Test finalization updates tracer components correctly"""
        sim.addcomponent_c(name="tracer_dust", gas_value=0.0, mu=0.0, dust_value=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dust_tracer=True)
        sim.addcomponent_c(name="tracer_gas", gas_value=np.array([10.0, 20.0, 30.0]), mu=18.0, gas_tracer=True)
        sim.update()
        sim.run()

        # assert tracer values are updated correctly
        np.testing.assert_array_almost_equal(sim.components.tracer_dust.dust.value,(sim.components.tracer_dust._Y/sim.dust._Y[:sim.grid._Nm_short*sim.grid.Nr]).reshape(sim.components.tracer_dust.dust.value.shape))
        np.testing.assert_array_almost_equal(sim.components.tracer_gas.gas.value,sim.components.tracer_gas._Y/ sim.gas.Sigma)

    def test_finalize_updates_active(self, sim):
        """Test finalization updates active components correctly"""
        sim.addcomponent_c(name="active_dust", gas_value=0.0, mu=0.0, dust_value=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), dust_active=True)
        sim.addcomponent_c(name="active_gas", gas_value=np.array([10.0, 20.0, 30.0]), mu=18.0, gas_active=True)
        sim.update()
        sim.run()

        # assert active values are updated correctly
        np.testing.assert_array_almost_equal(sim.components.active_dust.dust.Sigma,sim.components.active_dust._Y.reshape(sim.components.active_dust.dust.Sigma.shape))
        np.testing.assert_array_almost_equal(sim.components.active_gas.gas.Sigma,sim.components.active_gas._Y)

        #assert flags are reset
        assert sim.components._gas_updated == False
        assert sim.components._dust_updated == False


class TestJacobianCalculations:
    def test_Y_jacobian_basic(self, monkeypatch):
        """Test Y Jacobian calculation"""
        sim = Mock()
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        sim.grid.A = np.ones(4)
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        
        # Mock Jacobian matrices
        mock_gas_jac = sp.csc_matrix(np.eye(3))
        mock_dust_jac = sp.csc_matrix(np.eye(6))
        mock_compo_jac = sp.csc_matrix(np.eye(9))
        
        sim.dust.Sigma.jacobian = Mock(return_value=mock_dust_jac)
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_gas_jac)
        monkeypatch.setattr('tripod.std.compo.jacobian_compo', lambda *args, **kwargs: mock_compo_jac)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = Y_jacobian(sim, x, component=Mock())
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (9, 9)  # 3 (gas) + 6 (dust)
    
    def test_Y_jacobian_with_nans(self, monkeypatch):
        """Test Y Jacobian with NaN values"""
        sim = Mock()
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        
        # Create matrices with NaN values
        mock_gas_jac = sp.csc_matrix(np.array([[1, 0, 0], [0, np.nan, 0], [0, 0, 1]]))
        mock_dust_jac = sp.csc_matrix(np.eye(6))
        mock_compo_jac = sp.csc_matrix(np.eye(9))
        
        sim.dust.Sigma.jacobian = Mock(return_value=mock_dust_jac)
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_gas_jac)
        monkeypatch.setattr('tripod.std.compo.jacobian_compo', lambda *args, **kwargs: mock_compo_jac)
        
        x = Mock()
        x.stepsize = 0.1
        
        with pytest.raises(ValueError, match="Jacobian contains NaN values"):
            Y_jacobian(sim, x, component=Mock())
    
    def test_jacobian_compo(self, monkeypatch):
        """Test component Jacobian calculation"""
        sim = Mock()
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        
        comp = Mock()
        
        # Mock sublimation and condensation functions
        monkeypatch.setattr('tripod.std.compo.L_sublimation', 
                           lambda sim, comp: np.ones((3, 2)) * 0.1)
        monkeypatch.setattr('tripod.std.compo.L_condensation', 
                           lambda sim, comp, **kwargs: np.ones((3, 2)) * 0.2)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = jacobian_compo(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (9, 9)  # (Nr * Nm_s) + Nr

class TestPhysicalProcesses:
    def test_A_grains_q_minus_4(self):
        """Test grain surface area calculation with q = -4"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.qrec = np.ones(3) * (-4.0)
        sim.dust.rhos = np.ones((3, 3)) * 1000.0  # Note: shape (3, 3) for indexing
        
        result = A_grains(sim)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_A_grains_q_minus_3(self):
        """Test grain surface area calculation with q = -3"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.qrec = np.ones(3) * (-3.0)
        sim.dust.rhos = np.ones((3, 3)) * 1000.0
        
        result = A_grains(sim)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_A_grains_general_q(self):
        """Test grain surface area calculation with general q values"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.qrec = np.ones(3) * (-3.5)
        sim.dust.rhos = np.ones((3, 3)) * 1000.0
        
        result = A_grains(sim)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_L_condensation(self, monkeypatch):
        """Test condensation rate calculation"""
        sim = Mock()
        sim.gas.Hp = np.ones(3) * 0.1
        sim.gas.T = np.ones(3) * 100.0
        
        comp = Mock()
        comp.gas.pars.mu = 18.0  # Water molecular weight
        
        monkeypatch.setattr('tripod.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result = L_condensation(sim, comp, Pstick=1.0)
        
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_L_condensation_with_pstick(self, monkeypatch):
        """Test condensation with different sticking probability"""
        sim = Mock()
        sim.gas.Hp = np.ones(3) * 0.1
        sim.gas.T = np.ones(3) * 100.0
        
        comp = Mock()
        comp.gas.pars.mu = 18.0
        
        monkeypatch.setattr('tripod.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result_full = L_condensation(sim, comp, Pstick=1.0)
        result_half = L_condensation(sim, comp, Pstick=0.5)
        
        np.testing.assert_array_almost_equal(result_half, result_full * 0.5)
    
    def test_L_sublimation_active(self, monkeypatch):
        """Test sublimation rate for active dust component"""
        sim = Mock()
        sim.gas.T = np.ones(3) * 150.0
        
        comp = Mock()
        comp.dust._tracer = False
        comp.dust._active = True
        comp.dust.Sigma = np.ones((3, 2)) * 5.0
        comp.gas.pars.mu = 18.0
        comp.gas.pars.nu = 1e13
        comp.gas.pars.Tsub = 100.0
        
        monkeypatch.setattr('tripod.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result = L_sublimation(sim, comp, N_bind=1e15)
        
        assert result.shape == (3, 2)
        assert np.all(result >= 0)
    
    def test_L_sublimation_invalid_component(self, monkeypatch):
        """Test sublimation with invalid component type"""
        sim = Mock()
        
        comp = Mock()
        comp.dust._tracer = False
        comp.dust._active = False  # Invalid combination
        
        monkeypatch.setattr('tripod.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        with pytest.raises(RuntimeError, match="Component dust type not recognized"):
            L_sublimation(sim, comp)
    
    def test_L_sublimation_thin_layer(self, monkeypatch):
        """Test sublimation with thin ice layer (N_layer < 1e-2)"""
        sim = Mock()
        sim.gas.T = np.ones(3) * 150.0
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        
        comp = Mock()
        comp.dust._tracer = True
        comp.dust._active = False
        comp.dust.value = np.ones((3, 2)) * 1e-6  # Very small value for thin layer
        comp.gas.pars.mu = 18.0
        comp.gas.pars.nu = 1e13
        comp.gas.pars.Tsub = 100.0
        
        monkeypatch.setattr('tripod.std.compo.A_grains', 
                           lambda sim: np.ones((3, 2)) * 1e-4)
        
        result = L_sublimation(sim, comp, N_bind=1e15)
        
        assert result.shape == (3, 2)
        assert np.all(result >= 0)

class TestComponentJacobians:
    def test_c_jacobian_gas_active(self, monkeypatch):
        """Test component Jacobian for gas active component"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = False
        comp.gas._active = True
        comp.gas._tracer = False
        
        mock_jac = sp.csc_matrix(np.eye(3))
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripod.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (3, 3)
    
    def test_c_jacobian_gas_tracer(self, monkeypatch):
        """Test component Jacobian for gas tracer component"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = False
        comp.gas._active = False
        comp.gas._tracer = True
        
        mock_jac = sp.csc_matrix(np.eye(3))
        
        monkeypatch.setattr('dustpy.std.gas.jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripod.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
    
    def test_c_jacobian_dust_tracer(self, monkeypatch):
        """Test component Jacobian for dust tracer component"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = True
        comp.gas._active = False
        comp.gas._tracer = False
        
        mock_jac = sp.csc_matrix(np.eye(6))
        
        monkeypatch.setattr('tripod.std.dust.jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripod.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
    
    def test_c_jacobian_mixed_component(self, monkeypatch):
        """Test component Jacobian for mixed dust and gas component"""
        sim = Mock()
        comp = Mock()
        comp.dust._tracer = True
        comp.gas._active = True
        
        mock_jac = sp.csc_matrix(np.eye(9))
        
        monkeypatch.setattr('tripod.std.compo.Y_jacobian', lambda *args, **kwargs: mock_jac)
        monkeypatch.setattr('tripod.std.compo.set_boundaries_component', 
                           lambda sim, J, dt, comp: J)
        
        x = Mock()
        x.stepsize = 0.1
        
        result = c_jacobian(sim, x, component=comp)
        
        assert isinstance(result, sp.csc_matrix)
        assert result.shape == (9, 9)
    
    def test_c_jacobian_invalid_component(self):
        """Test component Jacobian with invalid component type"""
        sim = Mock()
        comp = Mock()
        comp.dust._active = False
        comp.dust._tracer = False
        comp.gas._active = False
        comp.gas._tracer = False  # Invalid combination
        
        x = Mock()
        x.stepsize = 0.1
        
        with pytest.raises(RuntimeError, match="Component type not recognized"):
            c_jacobian(sim, x, component=comp)

class TestBoundaryConditions:
    @pytest.fixture
    def mock_boundary_sim(self):
        """Create a mock simulation for boundary testing"""
        sim = Mock()
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        sim.gas.Sigma = np.ones(3) * 100.0
        sim.dust.Sigma = np.ones((3, 2)) * 10.0
        sim.dust._Y = np.ones(6) * 5.0
        sim.components._dust_updated = False
        sim.components._gas_updated = False
        sim._dust_compo = True
        return sim
    
    def test_set_boundaries_gas_active_val_condition(self, mock_boundary_sim):
        """Test boundary conditions for gas active component with val condition"""
        sim = mock_boundary_sim
        J = sp.csc_matrix(np.eye(3))
        dt = 0.1
        
        # Mock gas active component
        comp = Mock()
        comp.dust._tracer = False
        comp.dust._active = False
        comp.gas._active = True
        comp.gas._tracer = False
        comp._Y_rhs = np.zeros(3)
        comp._S = np.zeros(3)
        
        # Mock boundary conditions
        comp.gas.boundary.inner = Mock()
        comp.gas.boundary.inner.condition = "val"
        comp.gas.boundary.inner.value = 150.0
        
        comp.gas.boundary.outer = Mock()
        comp.gas.boundary.outer.condition = "val"
        comp.gas.boundary.outer.value = 50.0
        
        result = set_boundaries_component(sim, J, dt, comp)
        
        assert comp._Y_rhs[0] == 150.0
        assert comp._Y_rhs[2] == 50.0  # Nr-1

