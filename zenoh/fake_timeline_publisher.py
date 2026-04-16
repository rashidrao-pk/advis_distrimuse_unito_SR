from __future__ import annotations

import argparse
import random
import time
from collections import OrderedDict, deque
from typing import Iterable, List

import msgpack
import zenoh

ALL_SAFETY_AREAS = ["PLeft", "PRight", "RoboArm", "ConvBelt"]


def ordered_area_list(areas: Iterable[str]) -> List[str]:
    order_map = {name: i for i, name in enumerate(ALL_SAFETY_AREAS)}
    return sorted(list(areas), key=lambda x: order_map.get(x, 999))


def ordered_dict_of_lists(mapping) -> OrderedDict:
    return OrderedDict((k, list(mapping[k])) for k in ordered_area_list(mapping.keys()))


def serializable_results(results) -> OrderedDict:
    cleaned = OrderedDict()
    for area in ordered_area_list(results.keys()):
        src = results[area]
        cleaned[area] = {
            "score": float(src.get("score", 0.0)),
            "threshold": float(src.get("threshold", 1.0)),
            "norm_score": float(src.get("norm_score", 0.0)),
            "is_anomalous": bool(src.get("is_anomalous", False)),
            "status": str(src.get("status", "unknown")),
        }
    return cleaned


def pack_timeline_state(msg_id: int, score_history, latest_results) -> bytes:
    now = time.time()
    sec = int(now)
    nanosec = int((now - sec) * 1_000_000_000)
    payload = {
        "frame_meta": {
            "msg_id": int(msg_id),
            "corr_frame_id": "fake_timeline",
            "stamp": {"sec": sec, "nanosec": nanosec},
        },
        "score_history": ordered_dict_of_lists(score_history),
        "latest_results": serializable_results(latest_results),
    }
    return msgpack.packb(payload, use_bin_type=True)


def make_config(endpoint: str) -> zenoh.Config:
    return zenoh.Config.from_json5(
        f'''
    {{
      mode: "client",
      connect: {{
        endpoints: ["{endpoint}"]
      }}
    }}
    '''
    )


def next_norm_value(rng: random.Random, current: float) -> float:
    value = current + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.04:
        value += rng.uniform(0.35, 0.9)
    if value > 1.3:
        value -= rng.uniform(0.05, 0.2)
    return max(0.0, min(2.0, value))


def main() -> None:
    parser = argparse.ArgumentParser("Fake ADVIS timeline publisher")
    parser.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--zenoh-key", default="advis/vis/timeline/state")
    parser.add_argument("--rate", type=float, default=5.0, help="Publish rate in Hz")
    parser.add_argument("--history", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--zenoh-log-level", default="error")
    args = parser.parse_args()

    period = 1.0 / max(args.rate, 0.1)
    rng = random.Random(args.seed)

    score_history = OrderedDict(
        (area, deque(maxlen=args.history)) for area in ordered_area_list(ALL_SAFETY_AREAS)
    )
    latest_results = OrderedDict()
    current_values = OrderedDict((area, rng.uniform(0.2, 0.9)) for area in ordered_area_list(ALL_SAFETY_AREAS))

    zenoh.init_log_from_env_or(args.zenoh_log_level)
    config = make_config(args.zenoh_endpoint)

    with zenoh.open(config) as session:
        pub = session.declare_publisher(
            args.zenoh_key,
            encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        )

        msg_id = 0
        print(f"Publishing fake timeline data to {args.zenoh_key} via {args.zenoh_endpoint}")
        try:
            while True:
                msg_id += 1

                for area in ordered_area_list(current_values.keys()):
                    norm = next_norm_value(rng, current_values[area])
                    current_values[area] = norm
                    score_history[area].append(norm)
                    latest_results[area] = {
                        "score": norm,
                        "threshold": 1.0,
                        "norm_score": norm,
                        "is_anomalous": norm > 1.0,
                        "status": "UNEXPECTED" if norm > 1.0 else "normal",
                    }

                payload = pack_timeline_state(msg_id, score_history, latest_results)
                pub.put(payload)

                summary = " | ".join(
                    f"{area}={latest_results[area]['norm_score']:.2f}"
                    for area in ordered_area_list(latest_results.keys())
                )
                print(f"published #{msg_id}: {summary}")
                time.sleep(period)
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            try:
                pub.undeclare()
            except Exception:
                pass


if __name__ == "__main__":
    main()

