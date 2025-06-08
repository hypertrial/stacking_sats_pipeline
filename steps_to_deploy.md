# Steps to Deploy/Update PyPI Package

This guide covers the complete process for deploying and updating the `stacking-sats-pipeline` package on PyPI.

## Prerequisites

### 1. Install Required Tools
```bash
pip install --upgrade pip build twine
```

### 2. Create PyPI Account
- **Production PyPI**: https://pypi.org/account/register/

### 3. Generate API Token
For security, use API tokens instead of passwords:

**Production PyPI:**
1. Go to https://pypi.org/manage/account/token/
2. Create new token with scope "Entire account" 
3. Save the token (starts with `pypi-`)

### 4. Configure Authentication
Create `~/.pypirc` file:
```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE
```

## Deployment Steps

### Step 1: Update Version
Update version in `pyproject.toml`:
```toml
[project]
name = "stacking-sats-pipeline"
version = "0.0.1"  # Current version - increment for updates
description = "Hypertrial's Stacking Sats Library - Optimized Bitcoin DCA"
```

### Step 2: Update Personal Information (First Time Only)
Edit `pyproject.toml` and update:
```toml
authors = [
    {name = "Matt Faltyn", email = "matt@trilemmacapital.com"}
]
maintainers = [
    {name = "Matt Faltyn", email = "matt@trilemmacapital.com"}
]

[project.urls]
Homepage = "https://github.com/hypertrial/stacking_sats_pipeline"
Repository = "https://github.com/hypertrial/stacking_sats_pipeline"
```

### Step 3: Clean Previous Builds
```bash
# Remove old build artifacts
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/
```

### Step 4: Run Tests
```bash
# Install package in development mode
pip install -e .

# Run tests to ensure everything works
python -m pytest tests/ -v

# Test basic functionality
python -c "import stacking_sats_pipeline; print('Import successful')"
```

### Step 5: Build the Package
```bash
python -m build
```

This creates:
- `dist/stacking_sats_pipeline-X.X.X-py3-none-any.whl` (wheel)
- `dist/stacking-sats-pipeline-X.X.X.tar.gz` (source distribution)

### Step 6: Upload to Production PyPI
```bash
# Upload to Production PyPI
python -m twine upload dist/*

# Test installation from Production PyPI
pip install stacking-sats-pipeline

# Verify installation
python -c "import stacking_sats_pipeline; print(f'Version: {stacking_sats_pipeline.__version__}')"
stacking-sats --help
```

## Updating an Existing Package

### For Bug Fixes (Patch Version)
- `1.0.0` → `1.0.1`
- Change: Bug fixes, small improvements

### For New Features (Minor Version)  
- `1.0.0` → `1.1.0`
- Change: New features, backward compatible

### For Breaking Changes (Major Version)
- `1.0.0` → `2.0.0` 
- Change: Breaking API changes

### Update Process
1. **Update version** in `pyproject.toml`
2. **Update changelog** (if you have one)
3. **Commit changes** to git
4. **Tag the release**: `git tag v1.0.1 && git push origin v1.0.1`
5. **Follow deployment steps** above

## Troubleshooting

### Common Issues

**"File already exists" error:**
- You're trying to upload the same version again
- Increment version number in `pyproject.toml`

**Import errors after installation:**
- Check package structure matches `pyproject.toml`
- Verify all `__init__.py` files are present

**Missing dependencies:**
- Check `dependencies` list in `pyproject.toml`
- Test in fresh virtual environment

**CLI command not found:**
- Check `[project.scripts]` section in `pyproject.toml`
- Verify main function exists and is callable

### Testing in Clean Environment
```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install your package
pip install stacking-sats-pipeline

# Test functionality
python -c "import stacking_sats_pipeline; print('Success!')"
stacking-sats --help

# Clean up
deactivate
rm -rf test_env
```

## Automation (Optional)

### GitHub Actions Workflow
Create `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Quick Reference Commands

```bash
# Complete deployment in one go:
rm -rf dist/ build/ *.egg-info/
python -m build
python -m twine upload dist/*
```

## Security Notes

- ✅ Use API tokens instead of passwords
- ✅ Store tokens in `~/.pypirc` with restricted permissions (`chmod 600 ~/.pypirc`)
- ✅ Never commit tokens to version control
- ✅ Use different tokens for Test PyPI and Production PyPI
- ✅ Rotate tokens periodically 