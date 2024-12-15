"""
Automated testing script for running tests using pytest.

This script: Runs tests using pytest, ensuring that any issues are surfaced.
"""

import subprocess
import sys


def run_command(command, description=""):
    """
    Run a shell command and handle errors.

    Parameters:
    - command (str): The shell command to run.
    - description (str): Description of the action being performed.
    """
    try:
        if description:
            print(f"\n{description}")
        print(f"Running: {command}")
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)


def run_tests():
    """Run tests using pytest recursively through all test folders."""
    run_command(
        f"{sys.executable} -m pytest tests/ -v --maxfail=3 --disable-warnings",
        "Running tests with pytest in all test folders"
    )


def main():
    """
    Execute the automated testing process.
    """
    print("Starting automated testing process...")

    run_tests()

    print("\nAll steps completed successfully!")


if __name__ == "__main__":
    main()