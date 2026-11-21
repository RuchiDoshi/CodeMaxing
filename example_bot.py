"""Survival-first bot for mechfight/1.

Scoring is placement, so the plan is to reach the late game stronger than
anyone hiding: farm crates, drones and cubes while the circle is wide and
contact is rare, take the fights that are already even, run from the ones
that are not, and hold the inside edge of the circle the zone is closing to.
"""
import math

import mechfight as mf

# Rough metres for the three distance buckets a sound reports.
SOUND_DIST = {"near": 40.0, "mid": 110.0, "far": 220.0}
# Sounds that mean a mech is there. `crate` and `npc_gun` are not threats.
THREAT_SOUNDS = {"gun", "mortar", "explosion", "laser", "laser_charge",
                 "lightning", "jet", "boost", "step", "reload"}
# Which loose weapon is worth carrying, best first.
PICKUP_RANK = {"laser": 3, "lightning": 2, "mortar": 1}
# Upgrade ids we prefer, best first; matched as substrings.
UPGRADE_RANK = ("armor", "sensor", "regen", "hp", "health", "energy", "speed")

# Metres from a thing's `pos` up to the point worth aiming at.
AIM_Z = {"mech": 4.5, "npc": 1.75, "crate": 1.0, "cube": 0.5}


def _unit(dx, dy):
    m = math.hypot(dx, dy)
    return (dx / m, dy / m) if m > 1e-6 else (0.0, 0.0)


def _clamp_circle(px, py, cx, cy, rad):
    """Pull a point inside a circle, leaving it alone if it already is."""
    dx, dy = px - cx, py - cy
    d = math.hypot(dx, dy)
    if d <= rad or d < 1e-6:
        return px, py
    return cx + dx * rad / d, cy + dy * rad / d


class Bot:
    name = "codemax-v2"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.reach = cfg["loot"]["pickup_radius"]
        self.mortar_g = cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
        self.stat_cap = cfg.get("cubes", {}).get("stat_cap", 10)
        self.hurt_bearing = None
        self.hurt_t = -99.0
        self.prev_xy = None
        self.stuck = 0

    # -- helpers ---------------------------------------------------------

    def _threat_away(self, x, y, mechs, sounds):
        """Unit vector pointing away from everything that might shoot us."""
        ax = ay = 0.0
        for m in mechs:
            dx, dy = x - m["pos"][0], y - m["pos"][1]
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            w = 1.0 / max(d, 15.0)
            ax += dx / d * w
            ay += dy / d * w
        for s in sounds:
            if s.get("kind") not in THREAT_SOUNDS:
                continue
            b = s.get("bearing")
            if b is None:
                continue
            # `bearing` points toward the source, so we subtract to flee it.
            w = 0.5 / max(SOUND_DIST.get(s.get("dist"), 160.0), 15.0)
            ax -= math.cos(b) * w
            ay -= math.sin(b) * w
        return _unit(ax, ay)

    def _pick_upgrade(self, choices):
        best_i, best_r = 0, -1
        for i, c in enumerate(choices):
            cid = str(c.get("id", ""))
            rank = 0
            for j, key in enumerate(UPGRADE_RANK):
                if key in cid:
                    rank = len(UPGRADE_RANK) - j
                    break
            if rank > best_r:
                best_i, best_r = i, rank
        return best_i

    def _farm_target(self, x, y, obs, npcs, want_weapon):
        """Nearest thing worth walking to, as (rank, pos). Lower rank first."""
        best = None
        def consider(rank, pos, limit):
            nonlocal best
            d = math.dist(pos[:2], (x, y))
            if d > limit:
                return
            key = (rank, d)
            if best is None or key < best[0]:
                best = (key, pos)

        for c in obs.get("cubes", []):
            consider(0, c["pos"], 90.0)          # free, silent, instant
        if want_weapon:
            for p in obs.get("pickups", []):
                if p.get("kind") != "gun":
                    consider(1, p["pos"], 70.0)
        for c in obs.get("crates", []):
            consider(2, c["pos"], 80.0)          # 1 cube, maybe a weapon
        for n in npcs:
            consider(3, n["pos"], 70.0)          # 2 cubes, but it shoots back
        return best[1] if best else None

    # -- main ------------------------------------------------------------

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs.get("t", 0.0)
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        energy = me.get("energy", 0)
        cubes = me.get("cubes", 0)
        alive = obs.get("alive", 16)
        act: mf.Action = {"tick": obs["tick"]}

        # Remember which way the last hit came from: facing it avoids the
        # rear-armour multiplier.
        for e in obs.get("events", []):
            if e.get("type") == "damaged" and e.get("bearing") is not None:
                self.hurt_bearing = e["bearing"]
                self.hurt_t = t
        recently_hurt = (t - self.hurt_t) < 2.0

        mechs = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        npcs = [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]
        away = self._threat_away(x, y, mechs, obs.get("sounds", []))
        n_vis = len(mechs)
        nearest_d = min((math.dist(m["pos"][:2], (x, y)) for m in mechs), default=1e9)

        # Outnumbered or badly hurt, we leave -- unless they are already on
        # top of us, where running just hands them our back (x1.5 damage).
        cornered = nearest_d < 30.0
        fleeing = (hp_frac < 0.45 or n_vis >= 2) and not cornered
        w1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
        farming = (not fleeing and n_vis == 0 and hp_frac > 0.55
                   and alive > 6 and (cubes < self.stat_cap or w1 is None))

        # -- where to stand ------------------------------------------------
        zone = obs["zone"]
        cx, cy = zone["center"]
        radius = zone["radius"]
        nx, ny = zone["next_center"]
        nradius = zone["next_radius"]
        urgent = not me.get("in_zone", True)
        closing = t > zone.get("shrink_start", 1e9) - 10.0

        if urgent:
            goal = (cx, cy)
        else:
            # Hold the inside edge of the next circle, on whichever side we
            # are already on, nudged away from known threats. The centre is
            # where everyone collides, so we stay off it.
            ox, oy = _unit(x - nx, y - ny)
            if (ox, oy) == (0.0, 0.0):
                ox, oy = 1.0, 0.0
            bx, by = _unit(ox + away[0] * 1.5, oy + away[1] * 1.5)
            ring = max(6.0, nradius * 0.72)
            goal = (nx + bx * ring, ny + by * ring)
            if fleeing:
                goal = _clamp_circle(x + away[0] * 70, y + away[1] * 70,
                                     nx, ny, max(6.0, nradius * 0.88))
            elif farming:
                spot = self._farm_target(x, y, obs, npcs, w1 is None)
                if spot is not None:
                    goal = (spot[0], spot[1])
                    # Never farm our way out of the circle we have to be in.
                    if closing:
                        goal = _clamp_circle(goal[0], goal[1], nx, ny,
                                             max(6.0, nradius * 0.9))
            goal = _clamp_circle(goal[0], goal[1], cx, cy, max(6.0, radius - 12))

        gdx, gdy = goal[0] - x, goal[1] - y
        gdist = math.hypot(gdx, gdy)
        mvx, mvy = _unit(gdx, gdy)
        if gdist < 3.0 and not fleeing and not urgent:
            mvx = mvy = 0.0  # standing still is silent
        elif gdist < 30.0 and not farming and n_vis == 0 and not urgent:
            mvx, mvy = mvx * 0.4, mvy * 0.4  # under step_speed, so silent

        # Jump when we have been shoved against something for a while.
        if self.prev_xy is not None and (mvx or mvy):
            self.stuck = self.stuck + 1 if math.dist(self.prev_xy, (x, y)) < 0.35 else 0
        self.prev_xy = (x, y)
        if self.stuck >= 4:
            act["jump"] = True
            self.stuck = 0

        act["move"] = [mvx, mvy]
        act["boost"] = bool((urgent or fleeing) and energy > 25 and (mvx or mvy))

        # -- what to shoot ---------------------------------------------------
        w0 = me["weapons"][0] if me["weapons"] else None
        muzzle = z + me.get("stats", {}).get("muzzle_height", 7.0)
        gun_stats = (w0 or {}).get("stats", {})
        gun_speed = gun_stats.get("speed", 252.0)
        gun_range = gun_stats.get("range", 230.0)

        # A lone hunter is worth killing: it leaves the squad for good and
        # drops its cubes plus two. A pair is not.
        if fleeing:
            engage_r = 0.0
        elif n_vis == 1:
            engage_r = 130.0
        else:
            engage_r = 60.0

        shoot, kind, best = None, None, 1e9
        for m in mechs:
            d = math.dist(m["pos"], me["pos"])
            if d > engage_r:
                continue
            score = d + (m.get("hp") or 1400) * 0.06  # finish the wounded
            if score < best:
                shoot, kind, best = m, "mech", score
        if shoot is None and not fleeing and n_vis == 0:
            for c in obs.get("crates", []):
                d = math.dist(c["pos"][:2], (x, y))
                if d < 60.0 and d < best:
                    shoot, kind, best = c, "crate", d
            for n in npcs:
                d = math.dist(n["pos"], me["pos"])
                if d < 55.0 and d < best:
                    shoot, kind, best = n, "npc", d

        # -- where to point --------------------------------------------------
        tdist = 0.0
        if shoot is not None:
            tx, ty, tz = shoot["pos"]
            tvx, tvy, tvz = (shoot.get("vel") or [0.0, 0.0, 0.0])
            zoff = AIM_Z.get(kind, 1.0)
            tdist = math.dist((tx, ty, tz + zoff), (x, y, muzzle))
            lead = tdist / max(gun_speed, 1.0)  # bullets do not home
            px, py = tx + tvx * lead, ty + tvy * lead
            pz = tz + tvz * lead + zoff
            yaw = math.atan2(py - y, px - x)
            pitch = math.atan2(pz - muzzle, max(math.hypot(px - x, py - y), 0.01))
        elif recently_hurt and self.hurt_bearing is not None:
            yaw, pitch = self.hurt_bearing, 0.0  # present front armour
        else:
            yaw = math.atan2(mvy, mvx) if (mvx or mvy) else me["aim"][0]
            pitch = 0.0

        fire0 = False
        fire1 = False
        kind1 = (w1 or {}).get("kind")

        if shoot is not None:
            ammo = (w0 or {}).get("ammo", 0)
            in_gun_range = tdist <= gun_range * 0.9
            # The heavy weapons are for mechs; crates and drones get the gun.
            if kind == "mech" and kind1 == "mortar" and 30.0 < tdist < 200.0 and not in_gun_range:
                v = (w1 or {}).get("stats", {}).get("speed", 72.0)
                s = self.mortar_g * tdist / (v * v)
                if s <= 1.0:
                    pitch = 0.5 * math.asin(s)  # low arc
                    fire1 = True
            else:
                fire0 = bool(in_gun_range and ammo > 0)
                if kind1 == "lightning" and tdist <= 42.0 and energy >= 5:
                    fire1 = True  # cheap, and it hits crates too
                elif kind1 == "laser" and kind == "mech" and tdist <= 260.0 and not fleeing:
                    # Hold to charge, let go to fire. Release past half charge
                    # rather than letting it auto-fire two seconds later.
                    fire1 = ((w1 or {}).get("charge") or 0.0) < 0.55

        act["aim"] = [yaw, pitch]
        act["fire"] = [fire0, fire1]
        if w0 and w0.get("ammo", 99) <= 2 and shoot is None and not w0.get("reload"):
            act["reload"] = True

        # -- loot and upgrades -------------------------------------------------
        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"].get("choices", []))
        if w1 is None:
            here = [p for p in obs.get("pickups", [])
                    if p.get("kind") != "gun"
                    and math.dist(p["pos"][:2], (x, y)) <= self.reach - 0.3
                    and abs(p["pos"][2] - z) <= 3.5]
            if here:
                act["take"] = max(here, key=lambda p: PICKUP_RANK.get(p["kind"], 0))["id"]

        mode = "flee" if fleeing else ("farm" if farming else "hold")
        mf.draw.label(f"hp {me['hp']:.0f} cu {cubes} {mode} {n_vis}")
        return act


mf.run(Bot)
