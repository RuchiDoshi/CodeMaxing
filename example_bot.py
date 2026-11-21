"""Survival-first bot for mechfight/1.

Scoring is placement, not kills, so this bot plays to outlive the field:
it holds the inside edge of the circle the zone is closing to, keeps its
distance from the pack, breaks contact and regenerates when hurt, and
only shoots when a fight has already found it.
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
    name = "codemax-v3"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.reach = cfg["loot"]["pickup_radius"]
        self.mortar_g = cfg.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
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

    # -- main ------------------------------------------------------------

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs.get("t", 0.0)
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        energy = me.get("energy", 0)
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

        nearest_d = min((math.dist(m["pos"][:2], (x, y)) for m in mechs), default=1e9)
        threatened = bool(mechs) or recently_hurt
        cornered = nearest_d < 35.0
        fleeing = hp_frac < 0.5 and threatened and not cornered

        # -- where to stand ------------------------------------------------
        zone = obs["zone"]
        cx, cy = zone["center"]
        radius = zone["radius"]
        nx, ny = zone["next_center"]
        nradius = zone["next_radius"]
        urgent = not me.get("in_zone", True)

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
            goal = _clamp_circle(goal[0], goal[1], cx, cy, max(6.0, radius - 12))

        # Detour for loot only when nothing is pressing us.
        w1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
        if not fleeing and not urgent:
            cubes = obs.get("cubes", [])
            cube = min(cubes, key=lambda c: math.dist(c["pos"][:2], (x, y))) if cubes else None
            if cube is not None and math.dist(cube["pos"][:2], (x, y)) < 30.0:
                goal = (cube["pos"][0], cube["pos"][1])
            elif w1 is None:
                loot = [p for p in obs.get("pickups", []) if p.get("kind") != "gun"]
                if loot:
                    p = min(loot, key=lambda p: math.dist(p["pos"][:2], (x, y)))
                    if math.dist(p["pos"][:2], (x, y)) < 45.0:
                        goal = (p["pos"][0], p["pos"][1])

        gdx, gdy = goal[0] - x, goal[1] - y
        gdist = math.hypot(gdx, gdy)
        mvx, mvy = _unit(gdx, gdy)
        if gdist < 5.0 and not fleeing and not urgent:
            mvx = mvy = 0.0  # standing still is silent
        elif gdist < 30.0 and not threatened and not urgent:
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

        # -- who to shoot ---------------------------------------------------
        w0 = me["weapons"][0] if me["weapons"] else None
        muzzle = z + me.get("stats", {}).get("muzzle_height", 7.0)
        gun_stats = (w0 or {}).get("stats", {})
        gun_speed = gun_stats.get("speed", 252.0)
        gun_range = gun_stats.get("range", 230.0)

        # Never start a fight. Firing is the loudest thing we can do (the gun
        # carries 320 m to a squad that shares vision), so we shoot only what
        # is already on top of us or already shooting at us.
        if fleeing:
            engage_r = 0.0
        elif recently_hurt:
            engage_r = 80.0
        else:
            engage_r = 40.0
        if cornered:
            engage_r = max(engage_r, 60.0)

        target, best = None, 1e9
        for m in mechs:
            d = math.dist(m["pos"], me["pos"])
            if d > engage_r:
                continue
            score = d + (m.get("hp") or 1400) * 0.06  # finish the wounded
            if score < best:
                target, best = m, score
        # Drones are free cubes when the coast is clear.
        if target is None and not mechs and not fleeing and hp_frac > 0.6:
            for m in npcs:
                d = math.dist(m["pos"], me["pos"])
                if d < 60.0 and d < best:
                    target, best = m, d

        # -- where to point --------------------------------------------------
        tdist = 0.0
        if target is not None:
            tx, ty, tz = target["pos"]
            tvx, tvy, tvz = (target.get("vel") or [0.0, 0.0, 0.0])
            tdist = math.dist((tx, ty, tz + 4.5), (x, y, muzzle))
            lead = tdist / max(gun_speed, 1.0)  # bullets do not home
            px, py = tx + tvx * lead, ty + tvy * lead
            pz = tz + tvz * lead + 4.5
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

        if target is not None:
            ammo = (w0 or {}).get("ammo", 0)
            in_gun_range = tdist <= gun_range * 0.9
            # Mortar for anything far enough that the gun would just be noise.
            if kind1 == "mortar" and 30.0 < tdist < 200.0 and not in_gun_range:
                v = (w1 or {}).get("stats", {}).get("speed", 72.0)
                s = self.mortar_g * tdist / (v * v)
                if s <= 1.0:
                    pitch = 0.5 * math.asin(s)  # low arc
                    fire1 = True
            else:
                fire0 = bool(in_gun_range and ammo > 0)
                if kind1 == "lightning" and tdist <= 42.0 and energy >= 5:
                    fire1 = True
                elif kind1 == "laser" and tdist <= 260.0 and not fleeing:
                    # Hold to charge, let go to fire. Release past half charge
                    # rather than letting it auto-fire two seconds later.
                    fire1 = ((w1 or {}).get("charge") or 0.0) < 0.55

        act["aim"] = [yaw, pitch]
        act["fire"] = [fire0, fire1]
        if w0 and w0.get("ammo", 99) <= 2 and target is None and not w0.get("reload"):
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

        mf.draw.label(f"hp {me['hp']:.0f} {'flee' if fleeing else 'hold'} {len(mechs)}")
        return act


mf.run(Bot)
