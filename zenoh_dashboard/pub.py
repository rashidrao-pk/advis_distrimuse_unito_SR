import json
import time
from datetime import datetime, timezone

import zenoh

KEY = "demo/latest/msg"

CONFIG = zenoh.Config.from_json5("""
{
  mode: "client",
  connect: {
    endpoints: ["tcp/127.0.0.1:7447"]
  }
}
""")

def main():
    zenoh.init_log_from_env_or("error")

    with zenoh.open(CONFIG) as session:
        pub = session.declare_publisher(KEY)

        n = 0
        while True:
            n += 1
            msg = {
                "seq": n,
                "ts": datetime.now(timezone.utc).isoformat()
            }
            payload = json.dumps(msg)
            pub.put(payload)
            print(f"A sent: {payload}")
            time.sleep(1)

if __name__ == "__main__":
    main()
