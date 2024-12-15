# Setup Guide for ANAugment

Welcome to the setup guide for **ANAugment** (Advanced and Novel Augmentation). Follow the steps below to get started.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Git](https://git-scm.com/)
- [Python](https://www.python.org/) (version 3.6 or higher)
- [pip](https://pip.pypa.io/en/stable/installation/)

## Installation Steps

1. **Clone the repository:**

    ```sh
    git clone https://github.com/lunovian/an-augment.git
    ```

2. **Navigate to the project directory:**

    ```sh
    cd an_augment
    ```

3. **Create a virtual environment:**

    ```sh
    python -m venv venv
    ```

4. **Activate the virtual environment:**
    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```sh
        source venv/bin/activate
        ```

5. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

To start the project, run:

```sh
python main.py
```

## Building the Project

To create a distribution package, run:

```sh
python setup.py sdist bdist_wheel
```

## Testing

To run tests, use:

```sh
pytest
```

For further assistance, please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file or open an issue on GitHub.

Happy coding!
