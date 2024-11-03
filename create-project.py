import os

# Project folder structure
folders = [
    '.config',
    'src',
    'tests'
]

# Create main project directory
main_dir = os.getcwd()

# Create the subdirectories
for folder in folders:
    os.makedirs(os.path.join(main_dir, folder), exist_ok=True)

# Create the .gitignore file
with open(os.path.join(main_dir, '.gitignore'), 'w') as f:
    f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# VS Code
.vscode/
*.code-workspace

# CUDA
*.i
*.ii
*.gpu
*.ptx
*.cubin
*.fatbin""")

# Create the README.md file
with open(os.path.join(main_dir, 'README.md'), 'w') as f:
    f.write("""# GPU Test

A project to benchmark GPU performance using Hugging Face models.

## Prerequisites
- Python 3.x
- PyTorch
- Transformers
- CUDA-capable GPU""")

print('gpu-test folder structure created.')