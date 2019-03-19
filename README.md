# RCWA
Simple implementation of transfer matrix method and
rigorous coupled wave analysis

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
