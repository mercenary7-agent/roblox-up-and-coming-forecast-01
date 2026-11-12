#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01 entry: rank-diffusion Monte Carlo forecaster.

Reads /input/history.ndjson (chart reads, oldest first) and /input/task.json,
writes /output/forecast.json (or $MAKEFUN_OUTPUT_DIR). Standard library only,
deterministic (seeded RNG, no clock, no network).
"""
import json, math, os, random, sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HOUR = 3600.0
P_FLOOR_NOTE = 1e-4

# ---- tuned parameters ----
SIMS = 2000
CAND_N = 35
G_SCALE = 1.0
REV_SCALE = 1.0
H_SCALE = 2.0
SIGMA = 0.10
MAX_AGE_H = 24 * 28
NEW_FLOOR = 0.2
NEW_SCALE = 0.5
GHOST_JITTER = 0.3
GHOST_DRIFT = 0.01
GHOST_SIGMA = 0.2
P_MIN = 0.002
MAX_ROWS = 40
SEED = 12345
# regression fitted on the archive: dr24 = B0 + B1*g24 + B2*logrank (dr = lr_now - lr_future)
# v2: full-scale growth coefficient (G_SCALE=1.0, REV_SCALE=1.0), 2000 sims for stabler tails
B0 = -0.787
B_G = 0.380
B_R = 0.198


def parse_ts(s):
    # ISO 8601 UTC
    s = s.replace("Z", "+00:00")
    import datetime
    return datetime.datetime.fromisoformat(s).timestamp()


def load_history():
    path = os.path.join(INPUT_DIR, "history.ndjson")
    reads = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            uac = sorted(o.get("up_and_coming") or [], key=lambda g: g.get("rank", 10**9))
            reads.append({
                "t": parse_ts(o["at"]),
                "ids": [g["universe_id"] for g in uac],
                "rank_of": {g["universe_id"]: g.get("rank", 10**9) for g in uac},
                "play_of": {g["universe_id"]: (g.get("playing") or 1) for g in uac},
            })
    reads.sort(key=lambda r: r["t"])
    return reads


def forecast(reads, cut, places, horizon, params):
    # per-game series
    ser = {}
    for r in reads:
        if r["t"] > cut:
            continue
        for gid in r["ids"]:
            ser.setdefault(gid, []).append((r["t"], r["rank_of"][gid], r["play_of"][gid]))
    for k in ser:
        ser[k].sort()
    last = None
    for r in reads:
        if r["t"] <= cut:
            last = r
    games = {}
    for gid in last["ids"]:
        s = ser.get(gid, [])
        p0 = last["play_of"][gid]
        r0 = last["rank_of"][gid]

        def at(dt):
            best = None
            for row in s:
                if row[0] <= dt:
                    best = row
            return best
        a24 = at(cut - 24 * HOUR)
        a48 = at(cut - 48 * HOUR)
        g24 = math.log(p0 / a24[2]) if a24 else 0.0
        g48 = math.log(p0 / a48[2]) / 2.0 if a48 else g24
        gscore = 0.5 * g24 + 0.5 * g48
        first = s[0][0] if s else cut
        games[gid] = {"lr": math.log(max(r0, 0.5)), "g": gscore, "first": first}
    cands = sorted(games.items(), key=lambda kv: kv[1]["lr"])[: params["cand_n"]]
    entry_ranks = []
    seen = set()
    for r in reads:
        if r["t"] > cut:
            continue
        for gid in r["ids"]:
            if gid not in seen:
                seen.add(gid)
                entry_ranks.append(r["rank_of"][gid])
    recent_entries = sum(1 for gid, s in ser.items() if s[0][0] >= cut - 72 * HOUR)
    new_per_hour = max(params["new_floor"], recent_entries / 72.0) * params["new_scale"]
    rng = random.Random(params["seed"])
    sims = params["sims"]
    place_count = [dict() for _ in range(places)]
    ghost_lr = [math.log(max(e, 0.5)) for e in entry_ranks] or [math.log(50.0)]
    tot = sims * horizon
    for _ in range(sims):
        state = {gid: [g["lr"], g] for gid, g in cands}
        ghosts = []
        for h in range(1, horizon + 1):
            for gid in state:
                lr, g = state[gid]
                dr24 = params["b0"] + params["b_g"] * g["g"] * params["g_scale"] + params["b_r"] * lr * params["rev_scale"]
                drift = -dr24 / 24.0 * params["h_scale"]
                state[gid][0] = lr + drift + params["sigma"] * rng.gauss(0, 1)
            lam = new_per_hour
            if lam < 1:
                n_new = 1 if rng.random() < lam else 0
            else:
                n_new = int(lam) + (1 if rng.random() < lam - int(lam) else 0)
            for _ in range(n_new):
                ghosts.append([rng.choice(ghost_lr) + params["ghost_jitter"] * rng.gauss(0, 1)])
            for gid in list(state.keys()):
                age_h = (cut - state[gid][1]["first"]) / HOUR
                if age_h > params["max_age_h"] and rng.random() < min(1.0, (age_h - params["max_age_h"]) / 6.0):
                    del state[gid]
            items = [(state[gid][0], gid) for gid in state]
            items += [(gl[0], None) for gl in ghosts]
            items.sort(key=lambda x: x[0])
            for k in range(places):
                if k < len(items) and items[k][1] is not None:
                    gid = items[k][1]
                    place_count[k][gid] = place_count[k].get(gid, 0) + 1
            for gl in ghosts:
                gl[0] += -params["ghost_drift"] + params["ghost_sigma"] * rng.gauss(0, 1)
    rows = []
    top10 = []
    for k in range(places):
        lst = sorted(place_count[k].items(), key=lambda kv: -kv[1])
        kept = 0
        for gid, c in lst:
            p = c / tot
            if p < params["p_min"] and kept >= 5:
                break
            rows.append({"universe_id": gid, "place": k + 1, "p": p})
            kept += 1
            if kept >= params["max_rows"]:
                break
        if lst:
            top10.append(lst[0][0])
    for gid, g in sorted(games.items(), key=lambda kv: kv[1]["lr"]):
        if len(top10) >= places:
            break
        if gid not in top10:
            top10.append(gid)
    return {"probabilities": rows, "top10": top10[:places]}


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8"))
    places = int(task.get("places", 10))
    window = task.get("window", {})
    horizon = int(window.get("slots", 72))
    cut = parse_ts(task["history_until"])
    reads = load_history()
    params = {
        "sims": SIMS, "cand_n": CAND_N, "g_scale": G_SCALE, "b0": B0, "b_g": B_G,
        "b_r": B_R, "rev_scale": REV_SCALE, "h_scale": H_SCALE, "sigma": SIGMA,
        "max_age_h": MAX_AGE_H, "new_floor": NEW_FLOOR, "new_scale": NEW_SCALE,
        "ghost_jitter": GHOST_JITTER, "ghost_drift": GHOST_DRIFT,
        "ghost_sigma": GHOST_SIGMA, "p_min": P_MIN, "max_rows": MAX_ROWS,
        "seed": SEED,
    }
    fc = forecast(reads, cut, places, horizon, params)
    out = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fc, f)


if __name__ == "__main__":
    main()
