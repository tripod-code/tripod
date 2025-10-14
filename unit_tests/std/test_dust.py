import re
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from simframe.integration import Scheme

import scipy.sparse as sp

from tripod.std.dust import (
    dt, dt_Sigma, dt_smax, S_smax_hyd, S_hyd_compo, S_tot_compo,
    rhos_compo, Fi_sig1smax, dt_compo, prepare, finalize, 
    smax_initial, Sigma_initial, jacobian, a, F_adv, F_diff,
    m, p_frag, p_stick, H, rho_midplane, smax_deriv, S_coag,
    S_tot_ext, enforce_f, dadsig, dsigda, S_tot, S_compo,
    vrel_brownian_motion, q_eff, q_frag, q_rec, p_frag_trans,
    p_drift_frag, D_mod, vrad_mod, Y_jacobian, impl_1_direct
)

class TestDustTimesteps:
    def test_dt_basic(self, monkeypatch):
        """Test basic time step calculation"""
        sim = Mock()
        
        # Mock dt_Sigma and dt_smax
        monkeypatch.setattr('tripod.std.dust.dt_Sigma', lambda _: 100.0)
        monkeypatch.setattr('tripod.std.dust.dt_smax', lambda _: 50.0)
        
        result = dt(sim)
        assert result == 50.0
    
    def test_dt_Sigma_no_negative_sources(self):
        """Test dt_Sigma when no negative source terms exist"""
        sim = Mock()
        sim.dust.S.tot = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        result = dt_Sigma(sim)
        assert result == 1e100
    
    def test_dt_Sigma_with_negative_sources(self, monkeypatch):
        """Test dt_Sigma with negative source terms"""
        sim = Mock()
        sim.dust.S.tot = np.array([[1.0, -2.0], [3.0, -4.0], [5.0, 6.0]])
        sim.dust.Sigma = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        sim.dust.SigmaFloor = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        sim.dust.s.max = np.array([1.0, 2.0, 3.0])
        sim.dust.f.crit = 0.425
        sim.dust.s.sdot_coag = np.array([0.1, 0.2, 0.3])
        sim.dust.S.smax_hyd = np.array([0.01, 0.02, 0.03])
        
        # Mock dsigda function
        monkeypatch.setattr('tripod.std.dust.dsigda', lambda s: np.array([1.0, 2.0, 3.0]))
        
        result = dt_Sigma(sim)
        assert isinstance(result, float)
        assert result > 0
    
    def test_dt_smax(self):
        """Test smax time step calculation"""
        sim = Mock()
        sim.dust.S.tot = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sim.dust.Sigma = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        sim.dust.s.max = np.array([1.0, 2.0, 3.0])
        sim.dust.s.sdot_coag = np.array([0.1, 0.2, 0.3])
        sim.dust.S.smax_hyd = np.array([0.01, 0.02, 0.03])
        
        result = dt_smax(sim)
        assert isinstance(result, float)
        assert result > 0
    
    def test_dt_compo(self):
        """Test component time step calculation"""
        sim = Mock()
        
        # Mock component with active dust
        comp1 = Mock()
        comp1.dust._active = True
        comp1.dust.Sigma = np.array([[10.0, 20.0,30.0], [30.0, 40.0,50.0], [50.0, 60.0,70.0]])
        comp1.dust.S.tot = np.array([[1.0, -2.0,-3.0], [3.0, -4.0,5.0], [5.0, 6.0,-7.0]])
        
        sim.components.__dict__ = {'comp1': comp1}
        sim.dust.S.tot = np.array([[1.0, -2.0, 3.0], [3.0, -4.0, 5.0], [-5.0, 6.0, 7.0]])
        sim.dust.SigmaFloor = 0.1*np.ones((3,3))
        sim.dust.Sigma = np.ones((3,3))
        
        result = dt_compo(sim)
        assert result > 0
        assert result < 1e100

class TestParticleProperties:
    def test_a_calculation(self, monkeypatch):
        """Test particle size calculation"""
        sim = Mock()
        sim.dust.s.min = np.array([1e-4, 2e-4, 3e-4])
        sim.dust.s.max = np.array([1e-2, 2e-2, 3e-2])
        sim.dust.qrec = np.array([-3.5, -3.0, -2.5])
        sim.dust.f.dv = 0.4*np.ones(3)
        sim.grid._Nm_long = 5
        
        result = a(sim)

        assert result.shape == (3, 5)
        assert all(result[:,0] < result[:,2])  # a0 < a1 
        assert all(result[:,1] < result[:,2])  # fudge *a1 < a1
        assert all(result[:,0] < result[:,4])  # a0 < amax
        assert all(result[:,2] < result[:,4])  # a1 < amax
        
        
    
    def test_m_calculation(self, monkeypatch):
        """Test particle mass calculation"""
        sim = Mock()
        sim.dust.a = np.logspace(-4, -2, 5).reshape(1, 5) * np.ones((3, 1))
        sim.dust.rhos = np.ones((3, 5)) * 1000.0
        sim.dust.fill = np.ones((3, 5))
        
        result = m(sim)
        expected_result = (4/3) * np.pi * sim.dust.rhos * sim.dust.fill * sim.dust.a**3
        assert result.shape == (3, 5)
        np.testing.assert_array_almost_equal(result, expected_result)
    
    def test_H_calculation(self, monkeypatch):
        """Test dust scale height calculation"""
        sim = Mock()
        sim.gas.Hp = np.ones(3)
        sim.dust.St = np.ones((3, 5))
        sim.dust.delta.vert = np.ones((3, 5))
        
        with patch('dustpy.std.dust_f.h_dubrulle1995') as mock_h:
            mock_h.return_value = np.ones((3, 5))
            result = H(sim)
            assert result.shape == (3, 5)
            mock_h.assert_called_once()
    
    def test_rho_midplane_calculation(self):
        """Test midplane density calculation"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.H = np.ones((3, 5))
        
        result = rho_midplane(sim)
        assert result.shape == (3, 5)
        assert np.all(result > 0)

class TestFluxCalculations:
    def test_F_adv(self, monkeypatch):
        """Test advective flux calculation"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.v.rad_flux = np.ones((3, 3))
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        with patch('dustpy.std.dust_f.fi_adv') as mock_fi_adv:
            mock_fi_adv.return_value = np.ones((4, 2))
            result = F_adv(sim)
            assert result.shape == (4, 2)
            mock_fi_adv.assert_called_once()
    
    def test_F_diff(self, monkeypatch):
        """Test diffusive flux calculation"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.D = np.ones((3, 3))
        sim.gas.Sigma = np.ones(3)
        sim.dust.St = np.ones((3, 3))
        sim.dust.f.drift = 1.
        sim.dust.delta.rad = np.ones(3)
        sim.gas.cs = np.ones(3)
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        monkeypatch.setattr('tripod.std.dust_f.fi_diff_no_limit', 
                           lambda D, Sigma, gas_Sigma, St_drift, cs_term, r, ri: np.ones((4, 2)))
        
        result = F_diff(sim)
        assert result.shape == (4, 2)
        # Check boundary conditions
        np.testing.assert_array_equal(result[:1, :], 0.0)
        np.testing.assert_array_equal(result[-1:, :], 0.0)

class TestSourceTerms:
    def test_S_coag(self, monkeypatch):
        """Test coagulation source terms"""
        sim = Mock()
        sim.dust.a = np.ones((3, 5))
        sim.dust.v.rel.tot = np.ones((3, 5, 5))
        sim.dust.H = np.ones((3, 5))
        sim.dust.m = np.ones((3, 5))
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.s.min = np.ones(3)
        sim.dust.s.max = np.ones(3) * 10
        sim.dust.q.eff = np.ones(3) * (-3.5)
        sim.dust.SigmaFloor = np.ones((3, 2)) * 1e-10
        
        monkeypatch.setattr('tripod.std.dust_f.s_coag', 
                           lambda cross_sec, v_rel, H, m, Sigma, smin, smax, q_eff, SigmaFloor: np.ones((3, 2)))
        
        result = S_coag(sim)
        assert result.shape == (3, 2)
    
    def test_S_smax_hyd(self, monkeypatch):
        """Test hydrodynamic source terms for smax"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.s.max = np.ones(3)
        sim.dust.S.hyd = np.ones((3, 2))
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        monkeypatch.setattr('tripod.std.dust.Fi_sig1smax', lambda s: np.ones(4))
        with patch('dustpy.std.dust_f.s_hyd') as mock_s_hyd:
            mock_s_hyd.return_value = np.ones((3, 2))
            result = S_smax_hyd(sim)
            assert result.shape == (3,)
    
    def test_S_compo_no_components(self):
        """Test component source terms when no components exist"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        # No components attribute
        delattr(sim, 'components')
        
        result = S_compo(sim)
        np.testing.assert_array_equal(result, np.zeros((3, 2)))
    
    def test_S_compo_with_components(self):
        """Test component source terms with active components"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        
        # Mock components
        comp1 = Mock()
        comp1.dust.S_Sigma = np.ones((3, 2)) * 0.1
        comp2 = Mock()
        comp2.dust.S_Sigma = np.ones((3, 2)) * 0.2
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'comp1': comp1,
            'comp2': comp2,
            '_private': Mock()  # Should be ignored
        }
        
        result = S_compo(sim)
        expected = np.ones((3, 2)) * 0.3
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_S_tot_compo(self):
        """Test total source terms for components"""
        group = Mock()
        group.S.ext = np.ones((3, 2)) * 0.1
        group.S.hyd = np.ones((3, 2)) * 0.2
        group.S.coag = np.ones((3, 2)) * 0.3
        
        sim = Mock()
        
        result = S_tot_compo(sim, group)
        expected = np.ones((3, 2)) * 0.6
        np.testing.assert_array_almost_equal(result, expected)

class TestProbabilities:
    def test_p_frag(self, monkeypatch):
        """Test fragmentation probability"""
        sim = Mock()
        sim.dust.v.rel.tot = np.ones((3, 5, 5)) * 10.0
        sim.dust.v.frag = np.ones(3) * 5.0
        
        monkeypatch.setattr('tripod.std.dust_f.pfrag', 
                           lambda v_rel, v_frag: np.ones(3) * 0.5)
        
        result = p_frag(sim)
        np.testing.assert_array_equal(result, np.ones(3) * 0.5)
    
    def test_p_stick(self):
        """Test sticking probability"""
        sim = Mock()
        sim.dust.p.frag = np.ones(3) * 0.3
        
        result = p_stick(sim)
        np.testing.assert_array_almost_equal(result, np.ones(3) * 0.7)
    
    def test_p_frag_trans(self, monkeypatch):
        """Test fragmentation transition probability"""
        sim = Mock()
        sim.dust.St = np.ones((3, 5))
        sim.gas.alpha = np.ones(3)
        sim.gas.Sigma = np.ones(3)
        sim.gas.mu = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.pfrag_trans', 
                           lambda St, alpha, Sigma, mu: np.ones(3) * 0.4)
        
        result = p_frag_trans(sim)
        np.testing.assert_array_almost_equal(result, np.ones(3) * 0.4)
    
    def test_p_drift_frag(self, monkeypatch):
        """Test drift fragmentation probability"""
        sim = Mock()
        sim.dust.v.rel.rad = np.ones((3, 5, 5))
        sim.dust.v.rel.azi = np.ones((3, 5, 5))
        sim.dust.St = np.ones((3, 5))
        sim.gas.alpha = np.ones(3)
        sim.gas.Sigma = np.ones(3)
        sim.gas.mu = np.ones(3)
        sim.gas.cs = np.ones(3)
        sim.dust.p.fragtrans = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.pdriftfrag', 
                           lambda v_rad, v_azi, St, alpha, Sigma, mu, cs, p_trans: np.ones(3) * 0.6)
        
        result = p_drift_frag(sim)
        np.testing.assert_array_equal(result, np.ones(3) * 0.6)

class TestInitialConditions:
    def test_smax_initial_no_drift_limit(self):
        """Test initial smax calculation without drift limitation"""
        sim = Mock()
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.ini.dust.allowDriftingParticles = True
        sim.ini.dust.aIniMax = 1e-2
        
        result = smax_initial(sim)
        np.testing.assert_array_equal(result, np.ones(3) * 1e-2)
    
    def test_smax_initial_with_drift_limit(self, monkeypatch):
        """Test initial smax calculation with drift limitation"""
        sim = Mock()
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.ini.dust.allowDriftingParticles = False
        sim.ini.dust.aIniMax = 1e-2
        sim.ini.dust.d2gRatio = 0.01
        sim.gas.P = np.ones(3) * 100.0
        sim.gas.Sigma = np.ones(3) * 1000.0
        sim.gas.cs = np.ones(3) * 1000.0
        sim.dust.fill = np.ones((3, 2))
        sim.dust.rhos = np.ones((3, 2)) * 1000.0
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        sim.grid.OmegaK = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.interp1d', 
                           lambda ri, r, P: np.ones(4) * 100.0)
        
        result = smax_initial(sim)
        assert result.shape == (3,)
        assert np.all(result >= sim.dust.s.min * 1.5)
    
    def test_Sigma_initial(self):
        """Test initial surface density calculation"""
        sim = Mock()
        sim.dust.q.eff = np.ones(3) * (-3.5)
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.SigmaFloor = np.ones((3, 2)) * 1e-10
        sim.grid.Nr = 3
        sim.ini.dust.d2gRatio = 0.01
        sim.gas.Sigma = np.ones(3) * 100.0
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        
        result = Sigma_initial(sim)
        assert result.shape == (3, 2)
        assert np.all(result >= 0)
    
    def test_Sigma_initial_q_equals_minus_4(self):
        """Test initial surface density with q = -4"""
        sim = Mock()
        sim.dust.q.eff = np.ones(3) * (-4.0)  # Special case
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.SigmaFloor = np.ones((3, 2)) * 1e-10
        sim.grid.Nr = 3
        sim.ini.dust.d2gRatio = 0.01
        sim.gas.Sigma = np.ones(3) * 100.0
        sim .grid.r = np.array([1.0, 2.0, 3.0])
        
        result = Sigma_initial(sim)
        assert result.shape == (3, 2)
        assert np.all(result >= 0)

class TestBoundaryEnforcement:
    def test_enforce_f(self, monkeypatch):
        """Test fragmentation barrier enforcement"""
        sim = Mock()
        sim.dust.f.crit = 0.5
        sim.dust.Sigma = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        sim.dust.s.max = np.array([1.0, 2.0, 3.0])
        sim.dust.s.lim = np.array([0.5, 1.0, 1.5])
        sim.dust.qrec = Mock()
        sim.dust.qrec.update = Mock()
        
        # Mock components
        comp1 = Mock()
        comp1.dust._active = True
        comp1.dust.Sigma = np.array([[5.0, 10.0], [15.0, 20.0], [25.0, 30.0]])
        
        sim.components = Mock()
        sim.components.__dict__ = {'comp1': comp1}
        
        monkeypatch.setattr('tripod.std.dust.dadsig', lambda s: np.ones(3))
        
        # Should not raise an error
        enforce_f(sim)
        assert sim.dust.qrec.update.called
    
    def test_dadsig(self, monkeypatch):
        """Test dadsig calculation"""
        sim = Mock()
        sim.dust.s.lim = np.ones(3)
        sim.dust.qrec = np.ones(3)
        sim.dust.f.crit = 0.5
        sim.dust.s.max = np.ones(3) * 2.0
        sim.dust.s.min = np.ones(3) * 0.1
        sim.dust.Sigma = np.ones((3, 2))
        
        monkeypatch.setattr('tripod.std.dust_f.dadsig', 
                           lambda s_lim, qrec, f_crit, s_max, s_min, Sigma: np.ones(3))
        
        result = dadsig(sim)
        np.testing.assert_array_equal(result, np.ones(3))
    
    def test_dsigda(self, monkeypatch):
        """Test dsigda calculation"""
        sim = Mock()
        sim.dust.s.lim = np.ones(3)
        sim.dust.qrec = np.ones(3)
        sim.dust.f.crit = 0.5
        sim.dust.s.max = np.ones(3) * 2.0
        sim.dust.s.min = np.ones(3) * 0.1
        sim.dust.Sigma = np.ones((3, 2))
        
        monkeypatch.setattr('tripod.std.dust_f.dsigda', 
                           lambda s_lim, qrec, f_crit, s_max, s_min, Sigma: np.ones(3))
        
        result = dsigda(sim)
        np.testing.assert_array_equal(result, np.ones(3))
        
class TestPhysicalQuantities:
    def test_q_eff(self):
        """Test effective q calculation"""
        sim = Mock()
        sim.dust.q.frag = np.ones(3) * (-3.5)
        sim.dust.q.sweep = np.ones(3) * (-2.0)
        sim.dust.p.frag = np.ones(3) * 0.6
        
        result = q_eff(sim)
        expected = (-3.5) * 0.6 + (-2.0) * (1.0 - 0.6)
        np.testing.assert_array_almost_equal(result, np.ones(3) * expected)
    
    def test_q_frag(self, monkeypatch):
        """Test fragmentation q calculation"""
        sim = Mock()
        sim.dust.p.driftfrag = np.ones(3)
        sim.dust.v.rel.tot = np.ones((3, 5, 5))
        sim.dust.v.frag = np.ones(3)
        sim.dust.St = np.ones((3, 5))
        sim.dust.q.turb1 = np.ones(3)
        sim.dust.q.turb2 = np.ones(3)
        sim.dust.q.drfrag = np.ones(3)
        sim.gas.alpha = np.ones(3)
        sim.gas.Sigma = np.ones(3)
        sim.gas.mu = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.qfrag', 
                           lambda p_drift, v_rel, v_frag, St, q_turb1, q_turb2, q_drfrag, alpha, Sigma, mu: np.ones(3) * (-3.0))
        
        result = q_frag(sim)
        np.testing.assert_array_equal(result, np.ones(3) * (-3.0))
    
    def test_q_rec(self):
        """Test size distribution exponent calculation"""
        sim = Mock()
        sim.dust.Sigma = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sim.dust.s.min = np.array([1e-4, 2e-4, 3e-4])
        sim.dust.s.max = np.array([1e-2, 2e-2, 3e-2])
        
        result = q_rec(sim)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
    
    def test_vrel_brownian_motion(self, monkeypatch):
        """Test Brownian motion relative velocity"""
        sim = Mock()
        sim.gas.cs = np.ones(3)
        sim.dust.m = np.ones((3, 5))
        sim.gas.T = np.ones(3)
        
        monkeypatch.setattr('tripod.std.dust_f.vrel_brownian_motion', 
                           lambda cs, m, T: np.ones((3, 5)))
        
        result = vrel_brownian_motion(sim)
        assert result.shape == (3, 5)

class TestVelocityAndDiffusion:
    def test_D_mod(self, monkeypatch):
        """Test modified diffusivity calculation"""
        sim = Mock()
        sim.dust.delta.rad = np.ones(3)
        sim.gas.cs = np.ones(3)
        sim.grid.OmegaK = np.ones(3)
        sim.dust.St = np.ones((3, 5))
        sim.dust.f.drift = 0.8
        
        with patch('dustpy.std.dust_f.d') as mock_d:
            mock_d.return_value = np.ones((3, 5))
            result = D_mod(sim)
            
            # Check boundary conditions
            assert np.all(result[:1, :] == 0.0)
            assert np.all(result[-2:, :] == 0.0)
    
    def test_vrad_mod(self, monkeypatch):
        """Test modified radial velocity calculation"""
        sim = Mock()
        sim.dust.St = np.ones((3, 5))
        sim.dust.f.drift = 0.3
        sim.dust.v.driftmax = np.ones((3, 5))
        sim.gas.v.rad = np.ones(3)
        
        with patch('dustpy.std.dust_f.vrad') as mock_vrad:
            mock_vrad.return_value = np.ones((3, 5))
            result = vrad_mod(sim)
            assert result.shape == (3, 5)

class TestCompositionFunctions:
    def test_rhos_compo(self):
        """Test material density from composition"""
        sim = Mock()
        sim.gas.Sigma = np.ones(3)
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.SigmaFloor = np.ones((3, 2)) * 1e-10
        sim.dust.rhos = np.ones((3, 2)) * 1000.0
        
        # Mock components
        comp1 = Mock()
        comp1.dust._active = True
        comp1.dust.Sigma = np.ones((3, 2)) * 10.0
        comp1.dust.pars.rhos = 2000.0
        
        comp2 = Mock()
        comp2.dust._active = True
        comp2.dust.Sigma = np.ones((3, 2)) * 5.0
        comp2.dust.pars.rhos = 1500.0
        
        sim.components = Mock()
        sim.components.__dict__ = {
            'comp1': comp1,
            'comp2': comp2,
            '_private': Mock()
        }
        
        result = rhos_compo(sim)
        assert result.shape == (3, 2)
        assert np.all(result > 0)
    
    def test_S_hyd_compo(self, monkeypatch):
        """Test hydrodynamic source terms for components"""
        sim = Mock()
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        
        group = Mock()
        group.Fi = np.ones(4)
        
        with patch('dustpy.std.dust_f.s_hyd') as mock_s_hyd:
            mock_s_hyd.return_value = np.ones((3, 2))
            result = S_hyd_compo(sim, group)
            mock_s_hyd.assert_called_once_with(group.Fi, sim.grid.ri)
    
    def test_Fi_sig1smax(self, monkeypatch):
        """Test flux calculation for Sigma[1] * smax"""
        sim = Mock()
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.s.max = np.ones(3) * 2.0
        
        # Mock F_diff and F_adv to return proper shapes
        monkeypatch.setattr('tripod.std.dust.F_diff', 
                           lambda sim, Sigma: np.ones((4, 2)) * 0.1)
        monkeypatch.setattr('tripod.std.dust.F_adv', 
                           lambda sim, Sigma: np.ones((4, 2)) * 0.2)
        
        result = Fi_sig1smax(sim)
        expected = np.ones(4) * 0.3  # 0.1 + 0.2 for column 1
        np.testing.assert_array_almost_equal(result, expected)

class TestJacobianAndIntegration:
    @pytest.fixture
    def mock_sim_for_jacobian(self):
        """Create a mock simulation for Jacobian testing"""
        sim = Mock()
        sim.grid.Nr = 3
        sim.grid._Nm_short = 2
        sim.grid.r = np.array([1.0, 2.0, 3.0])
        sim.grid.ri = np.array([0.5, 1.5, 2.5, 3.5])
        sim.grid.A = np.ones(4)
        sim.dust.Sigma = np.ones((3, 2))
        sim.dust.a = np.ones((3, 5))
        sim.dust.v.rel.tot = np.ones((3, 5, 5))
        sim.dust.H = np.ones((3, 5))
        sim.dust.m = np.ones((3, 5))
        sim.dust.s.min = np.ones(3) * 1e-4
        sim.dust.s.max = np.ones(3) * 1e-2
        sim.dust.q.eff = np.ones(3) * (-3.5)
        sim.dust.D = np.ones((3, 5))
        sim.gas.Sigma = np.ones(3)
        sim.dust.v.rad_flux = np.ones((3, 5))
        sim.dust._rhs = np.zeros(6)  # Nr * Nm_short
        sim.dust.boundary.inner = None
        sim.dust.boundary.outer = None
        return sim
    
    def test_jacobian_basic(self, mock_sim_for_jacobian, monkeypatch):
        """Test basic Jacobian calculation"""
        sim = mock_sim_for_jacobian
        
        # Mock the jacobian generators
        sim.setattr('tripod.std.dust_f.jacobian_coagulation_generator',
                           lambda *args: (np.ones(10), np.arange(10), np.arange(10)))
        
        with patch('dustpy.std.dust_f.jacobian_hydrodynamic_generator') as mock_hyd_gen:
            mock_hyd_gen.return_value = (np.ones(6), np.arange(6), np.arange(6))
            x = Mock()
            x.stepsize = 0.1
            
            result = jacobian(sim, x)
            assert isinstance(result, sp.csc_matrix)
            assert result.shape == (6, 6)  # Nr * Nm_short
    

