import time
import csv
import math
import numpy as np
import pygame
from inputs import get_gamepad
import threading
import matplotlib.pyplot as plt

# Try to import GPIO for real Pi pinout
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# ============================================================
# GLOBAL SWITCHES & SETTINGS
# ============================================================
USE_HARDWARE = False      # True = real Pi GPIO
USE_XBOX = True           # Pilot control
LOG_TO_CSV = True
LOG_FILENAME = "odim_flight_log.csv"

# ============================================================
# 1. CORE MANIFOLD & TUNING (Physics Fix Integrated)
# ============================================================
A0, gamma, GM = 1.0, 0.5, 0.1
target_Theta = 2.0
dt = 0.05
Q_max = 5.0
MANIFOLD_TENSION = 1.0  # FIXED: Replaces the runaway 't' to keep math stable

# Simple "mass + drag" feel
CRAFT_MASS = 1.8          # kg-equivalent
DRAG_COEFF = 0.5         # generic drag factor

tuning = {
    "k_Q": 2.0,
    "k_F": 3.0,
    "thrust_scale": 1.0,      # Lift scaling
    "heave_scale": 5.0,
    "F_stab_gain": 0.2,       # auto-heave assist on F (only when bubble formed)
    "F_bubble_thresh": 0.06,  # |F| window for "formed"
    "Q_bubble_min": 0.2,
    "Q_bubble_max": 4.0,
    "F_health_scale": 3.0,
    "collapse_F_thresh": 1.5
}

# Option 2: Start on the ground, AUTO-HOVER rises to 3 ft and locks there
HOVER_Z_TARGET = 3.0        # Target altitude (ft-equivalent in sim units)
HOVER_KP_Z = 0.1
HOVER_KD_Z = 0.5
GROUND_Z = 0.0
GROUND_WARN_THRESH = 1.0

# State Vector: [t_local, x, y, z, vx, vy, vz, Q, yaw]
# Note: state[7] starts at 0.01 to avoid manifold 'Zero-Point' errors
# Start on the ground, at rest
state = [0.0, 0.0, 0.0, GROUND_Z, 0.0, 0.0, 0.0, 0.1, 0.0]

hw_map = {
    "custom_temp": 25.0,
    "phase_shift": 0.0,
    "clock_rate": 0.0,
    "xbox_heave": 0.0,   # A) Manual heave control (left stick up/down)
    "xbox_sway": 0.0,
    "yaw_input": 0.0,
    "pitch_input": 0.0,
    "voltage_rail": 12.6,
    "min_safe_volt": 10.5, # Critical cutoff for the 'Ashley Dawn' core
    "amps": 0.0,
    "watts": 0.0,
}

AUTO_HOVER_ENABLED = False
bubble_formed = False
bubble_health = 0.0
bubble_collapsed = False
bubble_recover_timer = 0.0
BUBBLE_RECOVER_TIME = 3.0

# ============================================================
# 2. DIAGNOSTICS & GPIO
# ============================================================
dtc_library = {
    "P0562": "12V Rail Voltage Low",
    "P0118": "Emitter Thermal Overload",
    "P1001": "Manifold Divergence (Collapse)"
}
active_codes = []
mil_status = False

def run_diagnostics(F_err, hw):
    global mil_status, active_codes, bubble_collapsed
    active_codes = []
    if abs(F_err) > tuning["collapse_F_thresh"]:
        active_codes.append("P1001")
        bubble_collapsed = True
    if hw["voltage_rail"] < 11.2:
        active_codes.append("P0562")
    if hw["custom_temp"] > 105.0:
        active_codes.append("P0118")
    mil_status = len(active_codes) > 0

# GPIO Setup
PIN_EMITTER_PWM = 13
PIN_CKP = 23
PIN_CMP = 24
_emitter_pwm = None

def setup_gpio():
    global _emitter_pwm
    if not (HAS_GPIO and USE_HARDWARE):
        return
    GPIO.setmode(GPIO.BCM)
    for pin in [PIN_EMITTER_PWM, PIN_CKP, PIN_CMP]:
        GPIO.setup(pin, GPIO.OUT)
    _emitter_pwm = GPIO.PWM(PIN_EMITTER_PWM, 1000)
    _emitter_pwm.start(0.0)

def update_gpio_outputs(hw, b_health, Q):
    if not (HAS_GPIO and USE_HARDWARE):
        return
    duty = 100.0 * np.clip(abs(Q) / Q_max, 0, 1) * b_health
    if _emitter_pwm:
        _emitter_pwm.ChangeDutyCycle(duty)
    rpm = max(hw["clock_rate"], 0.0)
    t_now = time.time()
    ckp_on = ((t_now * (rpm / 60.0) * 4) % 1.0) < 0.5
    cmp_on = ((t_now * (rpm / 60.0) * 2) % 1.0) < 0.5
    GPIO.output(PIN_CKP, GPIO.HIGH if ckp_on else GPIO.LOW)
    GPIO.output(PIN_CMP, GPIO.HIGH if cmp_on else GPIO.LOW)

# ============================================================
# POWER MODEL (12V Rail Simulation)
# ============================================================
# Virtual emitter electrical constants (simulation only)
EMITTER_RESISTANCE = 3.2      # ohms (virtual load)
EMITTER_INDUCTANCE = 0.004    # H (virtual)
BASE_IDLE_DRAW = 0.35         # amps (Pi + idle electronics)
THERMAL_RISE_RATE = 0.08      # degC per watt
THERMAL_COOL_RATE = 0.03      # degC per step
VOLTAGE_SAG_FACTOR = 0.015    # V drop per amp

def compute_power(hw, Q, bubble_health):
    """
    Returns (amps, watts) drawn from the 12V rail.
    This is a simulation-only model.
    """

    # Duty cycle proportional to |Q| and bubble health
    duty = np.clip(abs(Q) / Q_max, 0, 1) * bubble_health

    # Effective coil current (simulation only)
    # I = V * duty / R
    I_coil = (hw["voltage_rail"] * duty) / EMITTER_RESISTANCE

    # Total current draw
    I_total = BASE_IDLE_DRAW + I_coil

    # Power
    P_total = hw["voltage_rail"] * I_total

    # Thermal model
    hw["custom_temp"] += (P_total * THERMAL_RISE_RATE) - THERMAL_COOL_RATE
    hw["custom_temp"] = float(np.clip(hw["custom_temp"], 20.0, 140.0))

    # Voltage sag under load
    hw["voltage_rail"] = 12.6 - (I_total * VOLTAGE_SAG_FACTOR)
    hw["voltage_rail"] = float(np.clip(hw["voltage_rail"], 9.0, 12.6))

    return I_total, P_total

# ============================================================
# 3. CONTROLS (XBOX)
# ============================================================
def poll_gamepad():
    global AUTO_HOVER_ENABLED
    while True:
        try:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_X':
                    hw_map["yaw_input"] = event.state / 32768.0
                if event.code == 'ABS_Y':
                    # A) Manual heave control (left stick up/down)
                    hw_map["xbox_heave"] = -event.state / 32768.0
                if event.code == 'ABS_RX':
                    hw_map["xbox_sway"] = event.state / 32768.0
                if event.code == 'ABS_RY':
                    hw_map["pitch_input"] = -event.state / 32768.0
                if event.code == 'BTN_SOUTH' and event.state == 1:
                    # Toggle AUTO-HOVER: when enabled, climb to 3 ft and hold
                    AUTO_HOVER_ENABLED = not AUTO_HOVER_ENABLED
        except:
            time.sleep(0.01)

threading.Thread(target=poll_gamepad, daemon=True).start()

# ============================================================
# 4. PHYSICS & HUD
# ============================================================
def run_physics_step(s, hw, dt, tuning):
    global bubble_formed, bubble_health, bubble_collapsed, bubble_recover_timer
    t_loc, x, y, z, vx, vy, vz, Q, yaw = s

    # Stability using FIXED Manifold Tension
    r_mag = np.sqrt(x**2 + y**2 + z**2)
    r_mag = max(r_mag, 1.0)  # avoid huge F near origin
    g_r = 1.0 - (0.2 / r_mag)
    F = A0 * MANIFOLD_TENSION * g_r * (1.0 + gamma * Q) - target_Theta

    # Health Logic (only consider bubble "formed" once we're off the ground a bit)
    bubble_formed = (z > 0.2 and abs(F) < tuning["F_bubble_thresh"] and abs(Q) > 0.1)
    bubble_health = float(np.clip(1.0 - (abs(F) / tuning["F_health_scale"]), 0, 1))

    # Thrust Calculation (Absolute Lift)
    thrust = tuning["thrust_scale"] * abs(gamma * Q)

    # Auto-stabilization only when bubble is formed and not collapsed
    auto_stab = 0.0
    if bubble_formed and not bubble_collapsed:
        auto_stab = -tuning["F_stab_gain"] * F

    # Heave & AUTO-HOVER logic
    if AUTO_HOVER_ENABLED:
        # Simple PD altitude hold around HOVER_Z_TARGET
        z_err = HOVER_Z_TARGET - z
        heave_cmd = HOVER_KP_Z * z_err - HOVER_KD_Z * vz
    else:
        # Manual heave control (left stick up/down)
        heave_cmd = hw["xbox_heave"]

    # Add auto stabilization (only when bubble formed)
    heave_cmd += auto_stab

    # Recovery Mode
    if bubble_collapsed:
        bubble_recover_timer += dt
        heave_cmd += 0.4  # Forced emergency lift
        if bubble_recover_timer > BUBBLE_RECOVER_TIME:
            bubble_collapsed = False
            bubble_recover_timer = 0.0

    # Clamp heave command to a reasonable range
    heave_cmd = float(np.clip(heave_cmd, -1.0, 1.0))

    # Q-Dynamics (Targeting Stability)
    Q_target = heave_cmd * Q_max * 0.6 - (tuning["k_F"] * F)
    dQ = -tuning["k_Q"] * (Q - Q_target)

    # Simple drag model
    drag_x = -DRAG_COEFF * vx
    drag_y = -DRAG_COEFF * vy
    drag_z = -DRAG_COEFF * vz

    # Kinematics (forward/back, side-to-side visuals driven by x,y)
    dvx = (thrust * np.cos(yaw) + (hw["pitch_input"] * 2.0) + drag_x) / CRAFT_MASS
    dvy = (thrust * np.sin(yaw) + (hw["xbox_sway"] * 2.0) + drag_y) / CRAFT_MASS
    dvz = (thrust - 1.0 + (heave_cmd * tuning["heave_scale"]) + drag_z) / CRAFT_MASS

    # Update state
    new_s = [
        t_loc + dt,
        x + vx * dt,
        y + vy * dt,
        z + vz * dt,
        vx + dvx * dt,
        vy + dvy * dt,
        vz + dvz * dt,
        Q + dQ * dt,
        yaw + hw["yaw_input"] * 0.1
    ]

    # Ground clamp: no sinking below ground, no vertical velocity into ground
    if new_s[3] < GROUND_Z:
        new_s[3] = GROUND_Z
        new_s[6] = 0.0

    # Q clamp
    new_s[7] = float(np.clip(new_s[7], -Q_max, Q_max))

    # Virtual RPM for GPIO / HUD
    hw["clock_rate"] = 600 + 2000 * (abs(new_s[7]) / Q_max)
    return new_s, F

def draw_radar(screen, center, state, F, Q):
    cx, cy = center
    x = state[1]
    y = state[2]

    # Background disk
    pygame.draw.circle(screen, (5, 5, 25), center, 140)

    # emitter ring
    for i in range(8):
        angle = np.radians(i * 45) + state[8]
        ex = int(cx + 120 * np.cos(angle))
        ey = int(cy + 120 * np.sin(angle))
        color = (0, 200, 255) if (int(time.time() * 10) % 8 == i) else (20, 40, 80)
        pygame.draw.circle(screen, color, (ex, ey), 10, 0 if color[0] == 0 else 2)

    # bubble radius
    b_rad = int(40 * (1 + abs(Q) / Q_max))
    pygame.draw.circle(
        screen,
        (0, 255, 100) if abs(F) < 0.4 else (255, 100, 0),
        center,
        b_rad,
        2
    )

    # Forward/back / side-to-side visual: craft position marker
    # Scale x,y into radar space
    pos_scale = 15.0
    px = int(cx + x * pos_scale)
    py = int(cy - y * pos_scale)
    pygame.draw.circle(screen, (255, 255, 0), (px, py), 6)

# ============================================================
# 5. QUIET SCALAR ENGINE – FLIGHT LOG PLOTS
# ============================================================
def plot_flight_log(filename):
    t_vals, z_vals, Q_vals, F_vals = [], [], [], []
    try:
        with open(filename, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_vals.append(float(row["time"]))
                z_vals.append(float(row["z"]))
                Q_vals.append(float(row["Q"]))
                F_vals.append(float(row["F"]))
    except FileNotFoundError:
        print(f"[WARN] Flight log '{filename}' not found; skipping plots.")
        return
    except Exception as e:
        print(f"[WARN] Error reading flight log: {e}")
        return

    if len(t_vals) == 0:
        print("[WARN] Flight log is empty; skipping plots.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("ODIM / Quiet Scalar Engine – Flight Log")

    # z(t)
    axes[0].plot(t_vals, z_vals, 'b-', label="z")
    axes[0].set_ylabel("z")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    # Q(t)
    axes[1].plot(t_vals, Q_vals, 'orange', label="Q")
    axes[1].set_ylabel("Q")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    # F(t)
    axes[2].plot(t_vals, F_vals, 'g-', label="F")
    axes[2].set_ylabel("F")
    axes[2].set_xlabel("t")
    axes[2].grid(True)
    axes[2].legend(loc="best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption("ODIM Flight Deck – Quiet Scalar Engine")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()
    setup_gpio()
    curr_state = state[:]

    f = None
    writer = None
    if LOG_TO_CSV:
        f = open(LOG_FILENAME, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["time", "z", "Q", "F", "volt", "temp", "amps", "watts"])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        curr_state, F_err = run_physics_step(curr_state, hw_map, dt, tuning)

        # Power consumption model
        amps, watts = compute_power(hw_map, curr_state[7], bubble_health)
        hw_map["amps"] = amps
        hw_map["watts"] = watts

        run_diagnostics(F_err, hw_map)
        update_gpio_outputs(hw_map, bubble_health, curr_state[7])

        if LOG_TO_CSV and writer is not None:
            writer.writerow([
                curr_state[0],
                curr_state[3],
                curr_state[7],
                F_err,
                hw_map["voltage_rail"],
                hw_map["custom_temp"],
                hw_map["amps"],
                hw_map["watts"]
            ])

        screen.fill((5, 5, 15))
        draw_radar(screen, (350, 400), curr_state, F_err, curr_state[7])

        # HUD Panel
        px, py = 700, 50
        pygame.draw.rect(screen, (20, 20, 40), (px, py, 450, 700))
        stats = [
            "--- ODIM FLIGHT DECK (V2.3) ---",
            f"ALTITUDE AGL:    {curr_state[3]:.2f} m",
            f"VEL Z:           {curr_state[6]:.2f} m/s",
            f"POS X/Y:         {curr_state[1]:.2f} / {curr_state[2]:.2f}",
            f"VEL X/Y:         {curr_state[4]:.2f} / {curr_state[5]:.2f}",
            f"SCALAR DENSITY:  {curr_state[7]:.3f} Q",
            f"STABILITY (F):   {F_err:.4f}",
            f"BUBBLE HEALTH:   {bubble_health*100:.1f} %",
            f"FLIGHT MODE:     {'AUTO-HOVER' if AUTO_HOVER_ENABLED else 'MANUAL'}",
            f"BUBBLE STATE:    {'FORMED' if bubble_formed else 'SEARCHING'}",
            "-------------------------------",
            f"VIRTUAL RPM:     {hw_map['clock_rate']:.0f}",
            f"EMITTER TEMP:    {hw_map['custom_temp']:.1f} C",
            f"VOLTAGE RAIL:    {hw_map['voltage_rail']:.2f} V",
            f"CURRENT DRAW:    {hw_map['amps']:.2f} A",
            f"POWER DRAW:      {hw_map['watts']:.2f} W",
            "-------------------------------",
            "ACTIVE CODES:"
        ]
        for i, txt in enumerate(stats):
            screen.blit(font.render(txt, True, (200, 220, 255)), (px + 20, py + 30 + i * 24))

        for i, code in enumerate(active_codes):
            screen.blit(
                font.render(f"> {code}: {dtc_library.get(code)}", True, (255, 80, 80)),
                (px + 20, py + 360 + i * 22)
            )

        if curr_state[3] < GROUND_WARN_THRESH:
            pygame.draw.rect(screen, (200, 0, 0), (50, 50, 260, 50))
            screen.blit(font.render("GROUND PROXIMITY", True, (255, 255, 255)), (60, 60))

        pygame.display.flip()
        clock.tick(20)

    if f is not None:
        f.close()
    if HAS_GPIO:
        GPIO.cleanup()
    pygame.quit()

    # After sim closes, show Quiet Scalar Engine plots from the flight log
    if LOG_TO_CSV:
        plot_flight_log(LOG_FILENAME)

if __name__ == "__main__":
    main()