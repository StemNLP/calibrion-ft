# calibrion-ft

## Publishing to PyPI with uv

### 1. Init (first time only)

```bash
uv init --lib calibrion
```
As a result, the package files will be stored under `./calibrion`

### 2. Build

```bash
cd calibrion
uv build
```

### 3. Publish to PyPI

```bash
export UV_PUBLISH_TOKEN="pypi-AgEIcHlwaS5vcmcC..."
uv publish 
```

### 4. Bump version (before next release)

```bash
uv version patch   # 0.0.1 -> 0.0.2
uv version minor   # 0.1.0 -> 0.2.0
uv version major   # 1.0.0 -> 2.0.0
```
