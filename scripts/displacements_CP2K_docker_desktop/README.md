# README for CP2K Automation Script

This script automates CP2K calculations using Docker containers. It reads `.xyz` files, generates CP2K input files, and runs calculations in parallel.

**Prerequisites:**

- **Docker:** Ensure Docker is installed and properly configured on your system.
- **Docker SDK for Python:** Install the Docker SDK by running:

  ```bash
  pip install docker
  ```

- **Python Modules:** The script uses Python modules such as `argparse`, `glob`, `re`, `signal`, `multiprocessing`, and `docker`. Make sure these modules are available in your Python environment.

**Important Instructions:**

- **Customize Input Parameters:**
  - Modify the input parameters in the `generate_input_file` function within the script to match your computational needs.
  - Adjust parameters such as:
    - `BASIS_SET_FILE_NAME`: e.g., `BASIS_MOLOPT_UZH`
    - `POTENTIAL_FILE_NAME`: e.g., `POTENTIAL_UZH`
    - `CHARGE`, `MULTIPLICITY`, `UKS`
    - `&MGRID` settings: `CUTOFF`, `NGRIDS`, `REL_CUTOFF`
    - `&QS` settings: `METHOD`, `EXTRAPOLATION_ORDER`
    - `PREFERRED_DIAG_LIBRARY` etc.
  - You can replace or add parameters specific to your system e.g. CP2K docker path (docker_cmd).

- **Provide Necessary `.inc` Files:**
  - Ensure all required `.inc` files (e.g., `scf.inc`, `XC.inc`, `subsys.inc`) are present and contain settings tuned for your system.
  - Place these files in the working directory or adjust the script to point to their locations.

- **Prepare Your Working Directory:**
  - Include all your `dof_*_disp_*.xyz` files in the directory.
  - Make sure `dof_0_disp_0.xyz` is present, as it is required for the initial wavefunction.

- **Adjust Resource Settings:**
  - When running the script, specify the appropriate number of CPUs, processes, threads, and the CP2K Docker image version using command-line arguments.

**Example Usage:**

```bash
python run_displacements_CP2K.py --cpus 64 --processes 2 --threads 2 --version 2024.3_openmpi_native_psmp
```

**Note:** Customize the script and input files according to your computational requirements to ensure accurate and efficient simulations.