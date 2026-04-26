# Releasing pyshrew

## What a release produces

Pushing a `v*` tag triggers two independent workflows:

| Workflow | Output |
|---|---|
| `build-wheels.yml` | Python wheels on PyPI (`pip install pyshrew`) |
| `release-cpp.yml` | Static lib + header tarballs on GitHub Releases (C++ consumers) |

---

## Prerequisites

**PyPI OIDC trusted publisher** — once before the first release:
1. [pypi.org](https://pypi.org) → Your account → Publishing → "Add a new pending publisher"
   - PyPI project name: `pyshrew`
   - GitHub owner: `ffgiardina`
   - Repository: `shrew`
   - Workflow filename: `build-wheels.yml`
   - Environment: `pypi`
2. GitHub repo: Settings → Environments → New environment → name it `pypi`

---

## Release workflow

### 1. Bump the version

Edit `pyproject.toml`:
```toml
[project]
version = "0.2.0" 
```

### 2. Test locally

```bash
pip install scikit-build-core pybind11
pip install --no-build-isolation -e .
python -c "import pyshrew; print(dir(pyshrew))"
```

### 3. Build a local wheel and smoke-test it

```bash
pip install cibuildwheel

# Linux wheel
cibuildwheel --platform linux --archs x86_64

# macOS wheel
cibuildwheel --platform macos --archs arm64
```

Install and verify the built wheel:
```bash
pip install wheelhouse/pyshrew-*.whl
python -c "import pyshrew; rv = pyshrew.RandomVariable(pyshrew.NormalDistribution(0.0, 1.0)); print(rv)"
```

Inspect the wheel contents — confirm `__init__.py`, the `.so`, and bundled NLopt are all present:
```bash
python -m zipfile -l wheelhouse/pyshrew-*.whl | grep -E "\.(so|dylib|py)$"
```

### 4. (Optional) Upload to TestPyPI

```bash
pip install twine
twine upload --repository testpypi wheelhouse/*.whl
pip install --index-url https://test.pypi.org/simple/ pyshrew
python -c "import pyshrew; print('ok')"
```

### 5. Tag and push

```bash
git add pyproject.toml
git commit -m "Bump version to v0.2.0"
git tag v0.2.0
git push origin main v0.2.0
```

Pushing the tag fires both CI workflows automatically:
- `build-wheels.yml` builds all platform/Python combinations and publishes to PyPI
- `release-cpp.yml` builds static lib tarballs and attaches them to a GitHub Release

### 6. Verify the release

```bash
pip install pyshrew==0.2.0
python -c "import pyshrew; print(pyshrew.__doc__)"
```

---

## Troubleshooting

**Wheel build fails in CI**: set `CIBW_BUILD_VERBOSITY=2` locally or check the Actions log. The most common causes are a missing system dep in `before-all` or a CMake configure error.

**`auditwheel` refuses the wheel**: the extension links against a library that isn't in the manylinux whitelist and wasn't bundled. Run `auditwheel show wheelhouse/*.whl` locally to see which library is causing the issue, then ensure it's installed in `before-all`.

**PyPI publish fails with "trusted publisher not configured"**: the OIDC setup in step 0 above was skipped or the environment name doesn't match exactly.
