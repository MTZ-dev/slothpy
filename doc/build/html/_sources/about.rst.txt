About
=====

**SlothPy** is a cutting-edge software package dedicated to computational
molecular magnetism. Developed by **Mikołaj Żychowicz**, with significant
contributions from **Hubert Dziełak** in plotting and exporting modules, SlothPy is
under continuous evolution to meet the growing demands and and advancements of the field.
It aims to become a general utility library containing all relevant routines for
the theoretical investigation of nanomagnets.

Core Features
-------------

- **Interactive Scripting:** Designed for interactive use via Jupyter Notebooks
  or terminal environments, SlothPy offers a user-friendly, scripting-like
  experience. SlothPy harnesses advanced threading and multiprocessing techniques
  to ensure efficient performance, even with complex simulations. For guidance on
  getting started with interactive scripting, see the :ref:`how-to-start` section.

- **Customizable Workflows:** The software's flexible architecture allows users
  to craft their own pipelines and automated processes within Python environment.
  Thanks to Just-In-Time (JIT) compilation, SlothPy ensures that all workflows are
  inherently fast and efficient, facilitating rapid data processing and analysis.
  Refer to the :ref:`how-to-start` section for more on setting up your workflows.

- **Autotune Module:** Unlock the power of your hardware with a unique autotune
  feature that optimizes the performance of your machine. It automatically tunes
  and selects the optimal number of threads and processes for your CPU, providing
  access to the full performance potential. This module also estimates job completion
  times under various settings, allowing for more efficient workflow planning.

- **HDF5 File Utilization:** Emphasizing the use of HDF5 files, SlothPy offers
  powerful data management capabilities. HDF5's speed, portability, and ease of
  integration make it an ideal choice for handling complex data structures
  efficiently in computational workflows.

- **Enhanced User Experience:** SlothPy enhances the user experience with custom
  error handling, an improved printout system, a comprehensive user manual, and
  extensive documentation. These features make it easier for users to understand,
  troubleshoot, and effectively utilize the software. For detailed guidance and
  best practices, users are encouraged to consult the :ref:`reference-manual` section,
  which offers in-depth insights into SlothPy’s capabilities.

- **Extensibility:** Easily integrate results from popular chemistry programs
  like MOLCAS and ORCA. We are constantly expanding compatibility – reach out
  to include your preferred software in future releases. SlothPy utilizes HDF5
  files as a robust framework for data management, enhancing the scalability and
  accessibility of computational data.

- **Highly Visual:** SlothPy provides built-in functions for convenient data
  visualization, enabling users to easily interpret complex molecular data. The
  software includes a variety of visualization tools tailored to the needs of
  molecular magnetism, facilitating insightful and intuitive analysis.

- **Licensing & Access:** SlothPy is distributed under the GPL-3.0 license.
  Explore and contribute to the source code on `GitHub <https://github.com/MTZ-dev/slothpy/>`_.
  Download the latest version from the `PyPi repository <https://pypi.org/project/slothpy/>`_.
  
Technology Stack
----------------

SlothPy is built using a robust stack of technologies and libraries, ensuring
high performance, flexibility, and a wide range of features:

- **NumPy:** Fundamental package for scientific computing in Python, used for
  efficient array data structures. Visit `NumPy <https://numpy.org/>`_.

- **Numba:** An open-source JIT compiler that translates Python and NumPy code
  into fast machine code. Explore `Numba <https://numba.pydata.org/>`_.

- **h5py:** Interface to the HDF5 binary data format, utilized for managing
  complex data structures. Learn more about `h5py <https://www.h5py.org/>`_.

- **Matplotlib:** A comprehensive library for creating static, animated, and
  interactive visualizations in Python. Check out `Matplotlib <https://matplotlib.org/>`_.

- **ThreadPoolCtl:** Used for controlling the number of threads used in native
  libraries. More on `ThreadPoolCtl <https://github.com/joblib/threadpoolctl>`_.

- **SciPy:** Python-based ecosystem of open-source software for mathematics,
  science, and engineering, specifically for linear algebra algorithms. Visit `SciPy <https://scipy.org/>`_.

- **PyQt5:** A comprehensive set of Python bindings for Qt application framework, used for
  developing the GUI elements in the plotting modules. Learn more about `PyQt5 <https://riverbankcomputing.com/software/pyqt/intro>`_.

- **Pandas:** A powerful data analysis and manipulation library, used for
  integrating pipelines and exporting data efficiently. Explore `Pandas <https://pandas.pydata.org/>`_.

- **Multiprocessing & Multithreaded Linear Algebra Libraries:** Utilizes Python's
  multiprocessing for parallel processing and OpenMP (OMP) for multithreaded
  linear algebra operations.

By leveraging these technologies, SlothPy provides a powerful and versatile
platform for molecular magnetism research.

Support and Feedback
--------------------

Encounter an issue or have a suggestion? File a report on our `GitHub Issues page
<https://github.com/MTZ-dev/slothpy/issues>`_ or contact Mikołaj Żychowicz
directly at mikolaj.zychowicz@uj.edu.pl.

We hope that SlothPy will become a useful tool for the community dedicated to advancing molecular
magnetism research. Join us in shaping its future by sending us everything from bug reports, requests,
and suggestions, to feedback and comments.