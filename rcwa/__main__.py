from rcwa.common import get_input
import rcwa.rcwa as rcwa
import rcwa.tmm as tmm

def rcwa():
    input_toml = get_input()
    rcwa.main(input_toml)

def tmm():
    input_toml = get_input()
    tmm.main(input_toml)
