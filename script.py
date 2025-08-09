# Typical running script for any project
import os
import subprocess
import sys

def run_command_silently(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode

def create_virtualenv(venv_dir="venv"):
    print("Setting up virtual environment, please wait...")
    if not os.path.exists(venv_dir):
        ret = run_command_silently([sys.executable, "-m", "venv", venv_dir])
        if ret != 0:
            print("Error: Failed to create virtual environment.")
            sys.exit(1)
    else:
        print(f"Using existing virtual environment in ./{venv_dir}")
    print("Virtual environment ready.")


def install_dependencies(venv_dir="venv", requirements_file="requirements.txt"):
    print("Installing dependencies, please wait...")
    pip_path = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin", "pip")
    ret = run_command_silently([pip_path, "install", "-r", requirements_file])
    if ret != 0:
        print("Error: Failed to install dependencies.")
        sys.exit(1)
    print("Dependencies installed.")

def upgrade_pip(venv_dir="venv"):
    #print("Upgrading pip, please wait...")
    pip_path = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin", "pip")
    ret = run_command_silently([pip_path, "install", "--upgrade", "pip"])
    if ret != 0:
        print("Error: Failed to upgrade pip.")
        sys.exit(1)
    #print("Pip upgraded.")

def run_main_script(venv_dir="venv"):
    if os.name == "nt":
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")

    command = [python_exe, "-u", "main.py"]

    #print(f"Running main.py with {pptx_path} ...")
    print("\n")
    env = os.environ.copy()
    env["PYTHON_FORCE_COLOR"] = "1"
    subprocess.check_call(command)

def main():
    # pptx_path = input("Enter the full path to your PowerPoint (.pptx) file: ").strip()

    # if not os.path.isfile(pptx_path) or not pptx_path.lower().endswith(".pptx"):
    #     print("Error: The path is not a valid .pptx file.")
    #     sys.exit(1)

    venv_dir = "venvc"

    create_virtualenv(venv_dir)
    upgrade_pip(venv_dir)
    install_dependencies(venv_dir)
    run_main_script(venv_dir)

if __name__ == "__main__":
    main()