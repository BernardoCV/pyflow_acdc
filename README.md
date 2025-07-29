<img src="docs/images/logo_dark.svg" align="right" width="200px">

# PyFlow ACDC
A python-based tool for the design and analysis of hybrid AC/DC grids


PyFlow ACDC is a program worked on by ADOreD Project 

This project has received funding from the European Union’s  Horizon Europe 
Research and Innovation programme under the Marie Skłodowska-Curie grant 
agreement No 101073554.

## Important

This project is experimental and under active development. Issue reports and contributions are very welcome.

## Installation

### For Users
To run examples, download the folder to your repository including the csv folders.

### For Developers
#### Initial Setup
1. Install Git if you haven't already:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get install git
   # For Windows: Download from https://git-scm.com/download/win
   ```

2. Clone the repository:
```bash
git clone https://github.com/BernardoCV/pyflow_acdc.git
cd pyflow_acdc
```

3. Install in development mode:
```bash
pip install -e .
```
This installs the package in "editable" mode, allowing you to modify the code without reinstalling.

#### Making Changes

1. Create a new branch for your changes:
```bash
git checkout -b new-branch-name
git push origin new-branch-name
```

2. To push your changes to the remote repository:
```bash
git add .
git commit -m "Description of your changes"
git pull origin new-branch-name
git push origin new-branch-name
```

3. To pull the latest changes from the remote repository:
```bash
git pull origin main
```

To merge your changes into the main branch please contact the repository owner.

For mapping you will need to install the following packages:
```bash
pip install folium
```

For OPF you will need to install the following packages:
```bash

pip install pyomo
conda install -c conda-forge ipopt

```

**Note:** `ipopt` is not available on PyPI and must be installed via conda-forge.

For OPF run in Linux for the TEP:
```bash

sudo apt update
sudo apt install coinor-libbonmin-dev
conda install -c conda-forge ipopt
conda install -c conda-forge coin-or-bonmin

```


For Dash you will need to install the following packages:
```bash
pip install dash

```
## Test

```bash
pyflow-acdc-test       
```
Flags
```bash      
--quick      # Quick tests only
--tep        #TEP tests only


--show-output # All tests with output


```
## Documentation
Online documentation can be found at:

https://pyflow-acdc.readthedocs.io/

To build the latest documentation of a branch, build it locally.

To build the documentation:
```bash
cd docs
pip install -r requirements.txt
.\make html
```

The documentation will be available in `docs/_build/html/index.html`
