"""Small analytical enrollment and kernel checks; no measured data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec=importlib.util.spec_from_file_location('domain_personal_probe',Path(__file__).parents[1]/'scripts/domain_personal_probe.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_whole_record_action_split():
    sequence=np.repeat([f'E01/S01/A{i:02}' for i in range(1,28)],37)
    subject=np.repeat('S01',len(sequence))
    ends=np.tile(np.arange(7,297,8),27)
    cal,test=probe.enrollment(sequence,subject,ends,'S01')
    assert len(cal)==60 and len(test)==925
    assert set(sequence[cal])=={'E01/S01/A02','E01/S01/A06'}
    assert set(sequence[cal]).isdisjoint(sequence[test])
    with pytest.raises(ValueError,match='Forbidden'):
        probe.enrollment(np.repeat('E04/S40/A02',37),np.repeat('S40',37),np.arange(7,297,8),'S40')


def test_linear_dual_matches_primal_intercept_and_uses_cal_stats():
    torch.set_num_threads(1)
    rng=np.random.default_rng(5)
    x,z,y=rng.normal(size=(7,4)),rng.normal(size=(3,4)),rng.normal(size=(7,17,2))
    c=probe.matrix_cache(x,z)
    prediction,_=probe.dual_prediction(c['gram'],c['cross'],y,.3,True)
    xn=(x-x.mean(0))/np.maximum(x.std(0),.1)
    zn=(z-x.mean(0))/np.maximum(x.std(0),.1)
    xn,zn=xn-xn.mean(0),zn-xn.mean(0)
    yc=y.reshape(7,-1)-y.reshape(7,-1).mean(0)
    w=np.linalg.solve(xn.T@xn+.3*np.eye(4),xn.T@yc)
    expected=(zn@w+y.reshape(7,-1).mean(0)).reshape(3,17,2)
    np.testing.assert_allclose(prediction,expected,atol=1e-12)
    np.testing.assert_allclose(c['mean'].numpy(),x.mean(0),atol=1e-12)


def test_constant_targets_and_gaussian_solve_remain_constant():
    x=np.arange(24,dtype=float).reshape(6,4)
    c=probe.matrix_cache(x,x[:2])
    y=np.full((6,17,2),.4)
    sigma=c['median_distance']
    p,_=probe.dual_prediction(torch.exp(-c['distance']/(2*sigma*sigma)),torch.exp(-c['cross_distance']/(2*sigma*sigma)),y,.1,False)
    np.testing.assert_allclose(p,.4,atol=1e-12)


def test_per_action_motion_and_shuffle_analytical_cases():
    rng=np.random.default_rng(17)
    y=rng.uniform(.1,.9,size=(37,17,2))
    sequence=np.repeat('E01/S01/A01',37)
    perfect=probe.scores(y,y,sequence)
    assert perfect['nje']==0 and perfect['pck05']==1
    assert perfect['shuffle_nje_minus_aligned']>0
    for item in perfect['motion'].values():
        assert item['r2']==pytest.approx(1,abs=1e-12)
    static=probe.scores(np.broadcast_to(y.mean(0),y.shape),y,sequence)
    for item in static['motion'].values():
        assert item['r2']==pytest.approx(0,abs=1e-12)
