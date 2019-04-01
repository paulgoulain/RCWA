import os
import subprocess as sp
import pytest
import toml
from test_common import make_source_dic

# TODO add docstring
def make_layer_dic(epsilon, thickness):
    return {'epsilon': epsilon, 'thickness': thickness}

def test_benchmark_1():
    sp.call('rm output.toml', shell=True)
    source_dic = make_source_dic(2, 0.00001, 0.00001, [1, 0], [0, 0])
    periodicity_dic = {'period_x': 1.75, 'period_y': 1.5, 'harmonics_x': 3, 'harmonics_y': 3}
    superstrate_dic = {'epsilon': 9.0}
    layer_1_dic = make_layer_dic(os.path.join(os.getcwd(), 'test', 'rcwa_epsilon_benchmark_1_and_2.csv'), 0.5)
    print(layer_1_dic)
    layer_2_dic = make_layer_dic(6, 0.3)
    substrate_dic = {'epsilon': 2.0}
    input_toml = {'layer': [layer_1_dic, layer_2_dic], 'source': source_dic,\
            'superstrate': superstrate_dic, 'substrate': substrate_dic,\
            'periodicity': periodicity_dic}
    input_toml_path = os.path.join(os.getcwd(), 'input.toml')
    with open(input_toml_path, 'w') as fid:
        toml.dump(input_toml, fid)
    
    sp.call('python rcwa.py ' + input_toml_path, shell=True)
    output_toml = toml.load(os.path.join(os.getcwd(), 'output.toml'))
    assert output_toml['R']['-1-1'] == 0
    assert output_toml['R']['0-1'] == 0.0032
    assert output_toml['R']['1-1'] == 0
    assert output_toml['R']['-10'] == 0.0065
    assert output_toml['R']['00'] == 0.0789
    assert output_toml['R']['10'] == 0.0065
    assert output_toml['R']['-11'] == 0
    assert output_toml['R']['01'] == 0.0035
    assert output_toml['R']['11'] == 0
    assert output_toml['R']['sum'] == 0.0986


def test_benchmark_2():
    sp.call('rm output.toml', shell=True)
    source_dic = make_source_dic(2, 60.0, 30.0, [0.70711, 0.0], [0.0, 0.70711])
    periodicity_dic = {'period_x': 1.75, 'period_y': 1.5, 'harmonics_x': 3, 'harmonics_y': 3}
    superstrate_dic = {'epsilon': 9.0}
    layer_1_dic = make_layer_dic(os.path.join(os.getcwd(), 'test', 'rcwa_epsilon_benchmark_1_and_2.csv'), 0.5)
    print(layer_1_dic)
    layer_2_dic = make_layer_dic(6, 0.3)
    substrate_dic = {'epsilon': 2.0}
    input_toml = {'layer': [layer_1_dic, layer_2_dic], 'source': source_dic,\
            'superstrate': superstrate_dic, 'substrate': substrate_dic,\
            'periodicity': periodicity_dic}
    input_toml_path = os.path.join(os.getcwd(), 'input.toml')
    with open(input_toml_path, 'w') as fid:
        toml.dump(input_toml, fid)
    
    sp.call('python rcwa.py ' + input_toml_path, shell=True)
    output_toml = toml.load(os.path.join(os.getcwd(), 'output.toml'))
    assert output_toml['R']['-1-1'] == 0
    assert output_toml['R']['0-1'] == 0
    assert output_toml['R']['1-1'] == 0
    assert output_toml['R']['-10'] == 0
    assert output_toml['R']['00'] == 0.0850
    assert output_toml['R']['10'] == 0.0025
    assert output_toml['R']['-11'] == 0
    assert output_toml['R']['01'] == 0.0011
    assert output_toml['R']['11'] == 0.0004
    assert output_toml['R']['sum'] == 0.0889

def test_benchmark_3():
    '''Reproducing example 1 from L. Li "New formulation of the Fourier modal
    method for crossed surface relief gratings" JOSA 1997'''
    sp.call('rm output.toml', shell=True)
    source_dic = make_source_dic(1, 0.00001, 0.00001, [1.0, 0.0], [0.0, 0.0])
    periodicity_dic = {'period_x': 2.5, 'period_y': 2.5, 'harmonics_x': 11, 'harmonics_y': 11}
    superstrate_dic = {'epsilon': 2.25}
    layer_1_dic = make_layer_dic(os.path.join(os.getcwd(), 'test', 'rcwa_epsilon_benchmark_3.csv'), 1)
    print(layer_1_dic)
    substrate_dic = {'epsilon': 1.0}
    input_toml = {'layer': [layer_1_dic], 'source': source_dic,\
            'superstrate': superstrate_dic, 'substrate': substrate_dic,\
            'periodicity': periodicity_dic}
    input_toml_path = os.path.join(os.getcwd(), 'input.toml')
    with open(input_toml_path, 'w') as fid:
        toml.dump(input_toml, fid)
    
    sp.call('python rcwa.py ' + input_toml_path, shell=True)
    output_toml = toml.load(os.path.join(os.getcwd(), 'output.toml'))
    assert output_toml['T']['-1-1'] == 0.1281
    assert output_toml['T']['-11'] == 0.1281
    assert output_toml['T']['1-1'] == 0.1281
    assert output_toml['T']['11'] == 0.1281
    assert output_toml['T']['02'] == 0.0545
    assert output_toml['T']['0-2'] == 0.0545
    assert output_toml['T']['20'] == 0.0389
    assert output_toml['T']['-20'] == 0.0389
    assert output_toml['T']['00'] == 0.1881
