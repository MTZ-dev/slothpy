.. _installation-guide:

Installation Guide
==================

SlothPy is designed for easy installation, offering flexible options to suit a variety of user preferences. You can install SlothPy directly using pip, either within a virtual environment (recommended) or system-wide.

.. note:: 
   SlothPy requires Python 3.10 or higher for installation, with Python 3.11 recommended for improved performance.

Install via pip
---------------

The simplest way to install SlothPy is via pip.

**Creating a Virtual Environment (Recommended):**

Using a virtual environment is a good practice as it manages dependencies effectively and keeps your workspace organized.

- **On Linux/macOS:**

  1. Open your terminal.
  2. Create a new virtual environment:

     .. code-block:: bash

        python3 -m venv SlothPy_Env

  3. Activate the virtual environment:

     .. code-block:: bash

        source SlothPy_Env/bin/activate

- **On Windows:**

  1. Open Command Prompt.
  2. Create a new virtual environment:

     .. code-block:: batch

        python -m venv SlothPy_Env

  3. Activate the virtual environment:

     .. code-block:: batch

        SlothPy_Env\Scripts\activate

**Installing SlothPy:**

With the virtual environment activated (or directly in your system), install SlothPy using pip:

.. code-block:: bash

   pip install slothpy

This will install SlothPy along with all necessary dependencies.

Install from GitHub Repository
------------------------------

For those who prefer to install directly from the source:

1. Clone the SlothPy repository:

   .. code-block:: bash

      git clone https://github.com/MTZ-dev/slothpy.git

2. Navigate to the cloned directory:

   .. code-block:: bash

      cd slothpy

3. Now you can install the package in editable mode:

   .. code-block:: bash

      pip install -e .

or just set up the environment by installing the requirements file:

   .. code-block:: bash

      cd slothpy
      pip install -r requirements.txt

.. note:: 
   When installing SlothPy from the GitHub repository this way, ensure your scripts are in the same directory as SlothPy to enable proper importing of the SlothPy modules.

You're all set! SlothPy is now installed and ready for use.

Getting Started
---------------

Once SlothPy is installed, dive into its features by visiting the :ref:`how-to-start` section. We're excited to see the innovative ways you'll use SlothPy in your molecular magnetism research!

Keep Your Version Up to Date
----------------------------

SlothPy is a dynamically evolving software, with frequent updates to enhance features and performance. To ensure you are always working with the latest advancements, we recommend regularly updating your installation.

To update SlothPy, use the following pip command:

.. code-block:: bash

   pip install -U slothpy

This command will upgrade SlothPy to the latest available version, along with any required dependencies, ensuring you have access to the most current tools and improvements.