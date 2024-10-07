#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import re
import docker
import signal
from multiprocessing import Pool

def generate_input_file(dof_number, disp_number):
    project_name = f'dof_{dof_number}_disp_{disp_number}'
    input_filename = f'{project_name}.inp'
    xyz_filename = f'dof_{dof_number}_disp_{disp_number}.xyz'

    if dof_number == 0 and disp_number == 0:
        restart_wfn_line = ''
    else:
        restart_wfn_line = '    WFN_RESTART_FILE_NAME dof_0_disp_0-RESTART.wfn'

    input_template = """
############################This section is for DFT######################
&FORCE_EVAL
  METHOD QS
  STRESS_TENSOR  ANALYTICAL

   &DFT
    BASIS_SET_FILE_NAME   BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME   POTENTIAL_UZH
{restart_wfn_line}
    CHARGE  0
    MULTIPLICITY 1
    UKS  F

    &MGRID
      CUTOFF 1800
      NGRIDS 5
      REL_CUTOFF 90
    &END MGRID

    &POISSON
      PERIODIC XYZ
    &END POISSON

    &QS
      METHOD GAPW
      EXTRAPOLATION_ORDER  4
    &END QS

    &SCF
     @INCLUDE scf.inc
    &END SCF

    &XC
     @INCLUDE XC.inc
    &END XC

    &PRINT
      &MOMENTS
       FILENAME
      &END
     &END

  &END DFT

  &SUBSYS
   @INCLUDE subsys.inc

    &TOPOLOGY
       COORD_FILE_NAME {xyz_filename}
       COORD_FILE_FORMAT  XYZ
       NUMBER_OF_ATOMS  -1
       MULTIPLE_UNIT_CELL  1 1 1
     &END TOPOLOGY

  &END SUBSYS

  &PRINT
    &FORCES
     FILENAME
     NDIGITS 12
    &END
   &END

&END FORCE_EVAL

&GLOBAL
  PROJECT {project_name}
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL MEDIUM
  WALLTIME 255000
&END GLOBAL
"""

    with open(input_filename, 'w') as f:
        f.write(input_template.format(project_name=project_name, xyz_filename=xyz_filename, restart_wfn_line=restart_wfn_line))

    return input_filename

def run_cp2k(input_file, output_file, mpi_processes, threads, cp2k_version, container_list):
    client = docker.from_env()
    container = None
    try:
        volumes = {
            os.getcwd(): {
                'bind': '/mnt',
                'mode': 'rw'
            }
        }

        user = f"{os.getuid()}:{os.getgid()}"

        # Prepare the command to run inside the container
        command = [
            'mpirun',
            '-bind-to', 'none',
            '-np', str(mpi_processes),
            '-x', f'OMP_NUM_THREADS={threads}',
            'cp2k',
            '-i', input_file
        ]

        # Run the container
        container = client.containers.run(
            image=f'cp2k/cp2k:{cp2k_version}',
            command=command,
            volumes=volumes,
            user=user,
            working_dir='/mnt',
            detach=True,
            shm_size='4g',
            name=f'cp2k_{input_file}',
            stdout=True,
            stderr=True
        )

        # Add container to the list for cleanup
        container_list.append(container)

        # Stream logs to the output file
        with open(output_file, 'wb') as outfile:
            for log in container.logs(stream=True):
                outfile.write(log)
                outfile.flush()

        exit_code = container.wait()['StatusCode']

        if exit_code != 0:
            print(f"Container exited with code {exit_code} for {input_file}")
            raise Exception(f"Container error with exit code {exit_code}")

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt caught in run_cp2k for {input_file}")
        if container:
            container.kill()
            container.remove(force=True)
        raise
    finally:
        if container in container_list:
            container_list.remove(container)
        client.close()

def process_dof_disp(args_tuple):
    dof_disp, container_list = args_tuple
    dof_number, disp_number = dof_disp
    project_name = f'dof_{dof_number}_disp_{disp_number}'
    input_file = generate_input_file(dof_number, disp_number)
    output_file = f'{project_name}.out'
    threads_per_process = args.threads
    mpi_processes = (args.cpus // args.processes) // threads_per_process
    cp2k_version = args.version
    try:
        run_cp2k(input_file, output_file, mpi_processes, threads_per_process, cp2k_version, container_list)
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught in process_dof_disp for {project_name}")
        raise  # Re-raise to propagate the exception

    # Remove temporary files except for dof_0_disp_0
    if not (dof_number == 0 and disp_number == 0):
        files_to_remove = glob.glob(f'{project_name}-RESTART.wfn*')
        for f in files_to_remove:
            if os.path.isfile(f):
                os.remove(f)
    print(f'Completed calculation for {project_name}')

def signal_handler(sig, frame):
    print("\nKeyboardInterrupt caught, terminating all containers...")
    for container in global_container_list:
        try:
            container.kill()
            container.remove(force=True)
        except Exception as e:
            print(f"Error stopping container: {e}")
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CP2K calculations in Docker containers.')
    parser.add_argument('--cpus', type=int, default=16, help='Total number of CPUs to use.')
    parser.add_argument('--processes', type=int, default=1, help='Number of concurrent processes (containers).')
    parser.add_argument('--threads', type=int, default=2, help='Number of OMP threads per process.')
    parser.add_argument('--version', type=str, required=True, help='CP2K Docker image version to use.')

    args = parser.parse_args()

    # Global container list to keep track of running containers
    global_container_list = []

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Get list of xyz files
    xyz_files = glob.glob('dof_*_disp_*.xyz')
    pattern = re.compile(r'dof_(-?\d+)_disp_(-?\d+)\.xyz')

    dof_disp_list = []

    for filename in xyz_files:
        match = pattern.match(filename)
        if match:
            dof_number = int(match.group(1))
            disp_number = int(match.group(2))
            dof_disp_list.append((dof_number, disp_number))

    if (0, 0) not in dof_disp_list:
        raise ValueError('dof_0_disp_0.xyz file is missing. It is required to proceed.')

    # Remove (0, 0) from list and process it first
    dof_disp_list.remove((0, 0))

    # Process dof_0_disp_0 first
    try:
        print('Processing dof_0_disp_0...')
        input_file = generate_input_file(0, 0)
        output_file = 'dof_0_disp_0.out'
        threads = args.threads
        mpi_processes = args.cpus // threads  # Use allocated CPUs
        cp2k_version = args.version
        run_cp2k(input_file, output_file, mpi_processes, threads, cp2k_version, global_container_list)
        print('Completed calculation for dof_0_disp_0')
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught during dof_0_disp_0 processing.")
        sys.exit(1)

    # Process remaining dof and disp in parallel
    if dof_disp_list:
        print('Processing remaining calculations in parallel...')
        try:
            # Prepare arguments for multiprocessing
            container_lists = [[] for _ in range(args.processes)]
            pool_args = [((dof_disp_list[i], container_lists[i % args.processes])) for i in range(len(dof_disp_list))]

            with Pool(processes=args.processes) as pool:
                pool.map(process_dof_disp, pool_args)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught, terminating pool and containers...")
            pool.terminate()
            pool.join()
            for container_list in container_lists:
                for container in container_list:
                    try:
                        container.kill()
                        container.remove(force=True)
                    except Exception as e:
                        print(f"Error stopping container: {e}")
            sys.exit(1)
    else:
        print('No other dof_disp combinations to process.')
