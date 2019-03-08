import os
import subprocess as sp

import pytest
import toml
# TODO add docstring
def make_layer_dic(epsilon, mu, thickness):
    return {'epsilon': epsilon, 'mu': mu, 'thickness': thickness}

def make_source_dic(wavelength, theta, phi, te_amplitude, tm_amplitude):
    return {'wavelength': wavelength, 'theta': theta, 'phi': phi,\
            'te_amplitude': te_amplitude, 'tm_amplitude': tm_amplitude}

def test_benchmark():
    sp.call('rm output.toml', shell=True)
    source_dic = make_source_dic(1, 57, 23, [1, 0], [0, 1])
    superstrate_dic = {'mu': 1.2, 'epsilon': 1.4}
    layer_1_dic = make_layer_dic(2, 1, 0.25)
    layer_2_dic = make_layer_dic(1, 3, 0.5)
    substrate_dic = {'mu': 1.6, 'epsilon': 1.8}

    input_toml = {'layer': [layer_1_dic, layer_2_dic], 'source': source_dic,\
            'superstrate': superstrate_dic, 'substrate': substrate_dic}
    input_toml_path = os.path.join(os.getcwd(), 'input.toml')
    with open(input_toml_path, 'w') as fid:
        toml.dump(input_toml, fid)

    # TODO abs path here and for output
    sp.call('python tmm.py ' + input_toml_path, shell=True)
    output_toml = toml.load('output.toml')
    assert output_toml['output']['ref_eff'] == 0.4403
    assert output_toml['output']['trn_eff'] == 0.5597
    assert output_toml['output']['ref_eff+trn_eff'] == 1
