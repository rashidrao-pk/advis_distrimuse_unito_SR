## Setup Zenoh:

[guide here](https://zenoh.io/docs/getting-started/installation/)

```bash
pixi add --pypi msgpack eclipse-zenoh

pixi run python -c \
  "import zenoh, msgpack; print('Zenoh and msgpack are available')"

brew tap eclipse-zenoh/homebrew-zenoh


```
