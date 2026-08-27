## SETUP for Mac systems

### 1. Install Pixi on macOS

Pixi should be installed globally, not inside a Conda environment.

Using the official installer:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Reload Zsh:

```bash
source ~/.zshrc
```

Verify:

```bash
pixi --version
pixi info
```

Alternatively, with Homebrew:

```bash
brew install pixi
pixi --version
```

Create a separate Conda environment

```bash
conda create -n distrimuse-mac python=3.11 -y
conda activate distrimuse-mac
```

- install tool

```bash
conda install -c conda-forge \
   git \
   cmake \
   ninja \
   pkg-config \
   opencv \
   numpy \
   pyyaml \
   -y
```

3. Clone the repositories on macOS

```bash
mkdir -p ~/Projects/DistriMuSe
cd ~/Projects/DistriMuSe

git clone https://github.com/smart-robotics/distrimuse-image-broadcaster.git
git clone https://github.com/rashidrao-pk/advis_distrimuse_unito_SR.git
```

```bash
cd distrimuse-image-broadcaster
pixi workspace platform add osx-arm64
```

```bash
pixi lock
```

3. Create a macOS CycloneDDS configuration

```bash
cat > cyclonedds-macos.xml <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS
    xmlns="https://cdds.io/config"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="
        https://cdds.io/config
        https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">

  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface
            autodetermine="true"
            priority="default"
            multicast="default"/>
      </Interfaces>
      <AllowMulticast>default</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
EOF
```

Set the configuration for the current terminal:

```bash
export CYCLONEDDS_URI="file://$(pwd)/cyclonedds-macos.xml"
```

```bash
nano pixi.toml
```

replace

```md
[activation.env]
CYCLONEDDS_URI = "file:///some/path/cyclonedds.xml"
```

with

```md
[activation.env]
CYCLONEDDS_URI = "file://${PIXI_PROJECT_ROOT}/cyclonedds-macos.xml"
```

---
