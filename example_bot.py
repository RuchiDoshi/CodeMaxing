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
# Real upgrade ids from hello.config.upgrades, ranked for a bot that wins by
# outliving people. Repair nanites comes first because it is the one that
# pays off our whole plan: regen x1.6 and regen_delay -1.5 s per stack turns
# breaking contact into a full heal in seconds instead of half a minute.
UPGRADE_ORDER = {uid: i for i, uid in enumerate((
    "nanites",       # regen_per_s x1.6, regen_delay -1.5 s  (max 2)
    "plating",       # max hp x1.15                          (max 3)
    "armor",         # -8% damage taken                      (max 3)
    "servos",        # walk and boost speed +8%              (max 3)
    "sensors",       # +25 m sight, half the bearing error   (max 2)
    "gun_velocity",  # faster bullets land more of our leads (max 2)
    "gun_rate",
    "gun_damage",
    "gun_mag",
    "reactor",
    "capacitor",
    "jets",
    "gyros",
))}


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

def _predict_enemy_position(enemy, horizon=1.5):
    """
    Estimate where an enemy will be after horizon seconds.
    """
    ex, ey, ez = enemy["pos"]
    vx, vy, vz = enemy.get("vel", [0.0, 0.0, 0.0])

    return (
        ex + vx * horizon,
        ey + vy * horizon
    )


def _projectile_dodge(x, y, z, me, projectiles, gravity):
    """
    Predict incoming projectile collisions and return:
    (evasion_direction_x, evasion_direction_y, danger).
    """
    self_vel = me.get("vel") or [0.0, 0.0, 0.0]
    sx, sy = self_vel[:2]

    best_danger = 0.0
    best_direction = (0.0, 0.0)

    for p in projectiles:
        px, py, pz = p["pos"]
        vx, vy, vz = p["vel"]

        rx, ry = px - x, py - y
        rvx, rvy = vx - sx, vy - sy

        speed_sq = rvx * rvx + rvy * rvy
        if speed_sq < 1e-6:
            continue

        t = -(rx * rvx + ry * rvy) / speed_sq

        if t < 0.05 or t > 0.9:
            continue

        cx = rx + rvx * t
        cy = ry + rvy * t
        distance = math.hypot(cx, cy)

        if distance > 7.0:
            continue

        fall = 0.5 * gravity * t * t if p.get("kind") == "mortar" else 0.0
        projectile_z = pz + vz * t - fall
        mech_z = z + self_vel[2] * t

        if projectile_z < mech_z - 1.0:
            continue
        if projectile_z > mech_z + 10.0:
            continue

        danger = (7.0 - distance) / 7.0
        danger *= (0.9 - t) / 0.85
        danger = max(0.0, min(1.0, danger))

        if danger <= best_danger:
            continue

        speed = math.hypot(vx, vy)
        if speed < 1e-6:
            continue

        nx, ny = -vy / speed, vx / speed
        side = vx * (y - py) - vy * (x - px)

        if abs(side) < 1e-4:
            if nx * sx + ny * sy < 0:
                nx, ny = -nx, -ny
        elif side < 0:
            nx, ny = -nx, -ny

        best_danger = danger
        best_direction = (nx, ny)

    return best_direction[0], best_direction[1], best_danger

def _choose_position(x, y, cx, cy, radius,
                     nx, ny, nradius, enemies):
    """
    Evaluate candidate positions in the current/next safe-zone
    intersection. Return the highest-scoring position.
    """

    margin = 10.0
    usable_next = max(1.0, nradius - margin)
    usable_current = max(1.0, radius - margin)

    candidates = [(nx, ny)]

    # Generate candidate positions at different distances
    # and angles around the next zone's centre.
    for fraction in (0.25, 0.50, 0.72):
        r = usable_next * fraction

        for i in range(24):
            angle = (2.0 * math.pi * i) / 24

            px = nx + math.cos(angle) * r
            py = ny + math.sin(angle) * r

            candidates.append((px, py))

    best_position = None
    best_score = -float("inf")

    for px, py in candidates:

        # Candidate must fit inside both circles.
        d_current = math.hypot(px - cx, py - cy)
        d_next = math.hypot(px - nx, py - ny)

        if d_current > usable_current:
            continue

        if d_next > usable_next:
            continue

        # Reward positions with room before either zone edge.
        current_margin = (
            usable_current - d_current
        ) / usable_current

        next_margin = (
            usable_next - d_next
        ) / usable_next

        zone_score = (
            0.75 * current_margin
            + 1.25 * next_margin
        )

        # Penalise positions near enemy mechs.
        enemy_risk = 0.0

        for enemy in enemies:
            ex, ey = enemy["pos"][:2]
            distance = math.hypot(px - ex, py - ey)

            risk = max(0.0, (100.0 - distance) / 100.0)
            enemy_risk += risk ** 2

        # Avoid travelling unnecessarily far.
        travel_distance = math.hypot(px - x, py - y)

        travel_cost = travel_distance / max(nradius, 1.0)

        # Combined strategic score.
        score = (
            2.0 * zone_score
            - 2.5 * enemy_risk
            - 0.35 * travel_cost
        )

        if score > best_score:
            best_score = score
            best_position = (px, py)

    # Fallback if the circles do not overlap sufficiently.
    if best_position is None:
        best_position = _clamp_circle(
            nx, ny, cx, cy, max(6.0, radius - margin)
        )

    return best_position


class Bot:
    name = "codemax-v18"

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
        """Lowest rank wins; anything unlisted sits behind everything listed."""
        best_i, best_r = 0, 10 ** 6
        for i, c in enumerate(choices):
            rank = UPGRADE_ORDER.get(str(c.get("id", "")), 999)
            if rank < best_r:
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
            # Get back inside the current safe zone first.
            goal = (cx, cy)

        else:
            # Evaluate possible strategic positions.
            goal = _choose_position(
                x, y,
                cx, cy, radius,
                nx, ny, nradius,
                mechs
            )

            # If health is low, prioritise escaping threats.
            if fleeing:
                goal = _clamp_circle(
                    x + away[0] * 70,
                    y + away[1] * 70,
                    nx, ny,
                    max(6.0, nradius * 0.88)
                )

            # Keep the destination inside the current zone.
            goal = _clamp_circle(
                goal[0], goal[1],
                cx, cy,
                max(6.0, radius - 12)
            )

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

        # -- jink ------------------------------------------------------
        # There is no lock-on: every incoming shot is a lead prediction made
        # against our current velocity, and a bullet needs R/252 s to arrive.
        # Reversing our lateral velocity mid-flight puts us roughly
        # 2 * v_lat * (t - tau) from where the shooter aimed -- metres at
        # 100 m, against a 2.5 m radius -- while ground_accel 58.5 lets us
        # swap direction in about a quarter second. Legs are independent of
        # the torso, so aim and fire are completely untouched by this.
        if threatened and not urgent:
            if mvx == 0.0 and mvy == 0.0:
                # Standing still gives the shooter zero lead error, which is
                # a free hit. Circle the nearest threat instead.
                if mechs:
                    n = min(mechs, key=lambda m: math.dist(m["pos"][:2], (x, y)))
                    ex, ey = _unit(n["pos"][0] - x, n["pos"][1] - y)
                    mvx, mvy = -ey, ex
                else:
                    mvx, mvy = math.cos(me["aim"][0]), math.sin(me["aim"][0])
            # Flip every 8 ticks (0.4 s): long enough to build real lateral
            # speed, short enough that a reversal lands inside most flights.
            sign = 1.0 if (obs["tick"] // 8) % 2 else -1.0
            mvx, mvy = _unit(mvx - mvy * sign, mvy + mvx * sign)

        # -- predictive projectile dodging --------------------------
        projectiles = obs.get("projectiles", [])

        dodge_x, dodge_y, danger = _projectile_dodge(
            x, y, z,
            me,
            projectiles,
            self.mortar_g
        )

        if danger > 0.0:
            # Blend evasive movement with our strategic direction.
            dodge_weight = 0.35 + 0.55 * danger

            mvx = (1.0 - dodge_weight) * mvx + dodge_weight * dodge_x
            mvy = (1.0 - dodge_weight) * mvy + dodge_weight * dodge_y

            mvx, mvy = _unit(mvx, mvy)

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

        # Retreating used to mean holding fire, which handed the chaser free
        # damage. Legs and torso are independent (§6), so we can walk away and
        # keep shooting backwards -- and keeping them in our front arc denies
        # the x1.5 rear multiplier at the same time.
        # Shooting more has won every time we have tried it and lost every
        # time we have cut it back, so engage as far as the gun actually
        # reaches rather than at an arbitrary 110 m.
        engage_r = 90.0 if fleeing else (200.0 if hp_frac > 0.5 else 45.0)
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
        elif mechs:
            # With no firing target we used to aim along our escape heading,
            # which puts a pursuer at exactly 180 deg -- the rear arc, where
            # hits land at x1.5. The front arc is 180 deg wide, so simply
            # looking at them removes the multiplier entirely.
            n = min(mechs, key=lambda mm: math.dist(mm["pos"][:2], (x, y)))
            yaw = math.atan2(n["pos"][1] - y, n["pos"][0] - x)
            pitch = 0.0
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
        # Walking into a fight part-loaded means a 2.4 s reload at about
        # 187 dps incoming -- roughly 450 damage for nothing. Top up whenever
        # the field is clear instead of waiting until the magazine is dry.
        if (w0 and not mechs and not w0.get("reload")
                and w0.get("ammo", 99) < gun_stats.get("mag", 12)):
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
