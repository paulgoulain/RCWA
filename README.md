# RCWA
Simple implementation of transfer matrix method (TMM) and
rigorous coupled wave analysis (RCWA) based on the notes from Computational Electromagnetics Course by Raymond Rumpf: https://empossible.net/academics/emp5337/ (formerly: http://emlab.utep.edu/ee5390cem.htm)


## Getting Started
Before running any calculations of your own it is recommended to run
the test suite which check the output of a few tmm and rcwa runs. 
Go to the root folder and execute
```
pytest
```

To run a tmm computatation execute
``` 
python tmm.py path-to-input-toml-file
```
Analogously, for a rcwa run
```
python rcwa.py path-to-input-toml-file
```
which will read the provided input files in .toml format.
### Dependencies

numpy, toml, pytest

## Authors

* **Gleb Siroki**

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
