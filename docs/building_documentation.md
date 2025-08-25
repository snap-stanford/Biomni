
# Building Project Documentation

To build the project's documentation locally, use the `Makefile` located in the `docs` directory.

## Prerequisites

First, ensure you have activated the `biomni_e1` conda environment. Then, install the required dependencies:

```bash
pip install sphinx sphinx-rtd-theme
```


## Build the Documentation

1.  Navigate to the `docs` directory:

    ```bash
    cd docs
    ```

2.  Run the `make html` command:

    ```bash
    make html
    ```

    This command will automatically generate the API documentation and build all HTML files.


## View the Documentation

Once the build is complete, you can find the generated documentation in the `docs/build/html` directory.

Open the `index.html` file in your browser to view it.
