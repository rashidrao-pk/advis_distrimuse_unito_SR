import json
import time
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

def on_sample(sample):
    text = sample.payload.to_string()
    try:
        msg = json.loads(text)
    except Exception:
        msg = text
    print(f"LIVE {sample.key_expr} => {msg}")

def main():
    zenoh.init_log_from_env_or("error")

    with zenoh.open(CONFIG) as session:
        # Optional: print the latest stored value first, if storage is enabled.
        got_any = False
        for reply in session.get(KEY):
            try:
                sample = reply.ok
                text = sample.payload.to_string()
                try:
                    msg = json.loads(text)
                except Exception:
                    msg = text
                print(f"GET  {sample.key_expr} => {msg}")
                got_any = True
            except Exception:
                pass

        if not got_any:
            print("GET  (no stored value yet)")

        print("Waiting for live updates...")
        with session.declare_subscriber(KEY, on_sample):
            while True:
                time.sleep(1)

if __name__ == "__main__":
    main()
