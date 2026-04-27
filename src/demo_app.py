"""
DQN CartPole - Interactive Demo App
===================================
A pygame-based application that loads the trained model
and shows it playing CartPole in real time. The application
also includes controls and live stats.

Controls:
    SPACE   -> Play/Pause
    R       -> Reset episode
    Q/ESC   -> Quit application
    
Usage:
    cd src
    python demo_app.py
"""

import os, sys
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pygame
import gymnasium as gym

from agent import DQNAgent

# ------------------------------------------------------------------ #
#  PATHS                                                             #
# ------------------------------------------------------------------ #

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH = os.path.join(BASE_DIR, "results", "dqn-cartpole.pth")

# ------------------------------------------------------------------ #
#  WINDOW & LAYOUT                                                   #
# ------------------------------------------------------------------ #

WIN_W, WIN_H    = 900, 600
SIM_W           = 620
PANEL_W         = WIN_W - SIM_W
SIM_H           = WIN_H
FPS             = 60

# ------------------------------------------------------------------ #
#  COLOURS                                                           #
# ------------------------------------------------------------------ #

C_BG_SIM     = (15,  17,  26)
C_BG_PANEL   = (20,  23,  35)
C_ACCENT     = (82,  183, 255)
C_ACCENT2    = (255, 165,  60)
C_GREEN      = (80,  220, 140)
C_RED        = (255,  80,  80)
C_TEXT       = (220, 225, 240)
C_TEXT_DIM   = (100, 110, 135)
C_TRACK      = (40,   45,  65)
C_CART       = (82,  183, 255)
C_CART_WHEEL = (50,  120, 200)
C_POLE       = (255, 165,  60)
C_DIVIDER    = (35,   40,  58)

# ------------------------------------------------------------------ #
#  SIMULATION SCALING                                                #
# ------------------------------------------------------------------ #

SCALE       = 100.0
CART_W      = 80
CART_H      = 30
WHEEL_R     = 10
POLE_W      = 8
POLE_LEN    = 150

# ------------------------------------------------------------------ #
#  APP STATES                                                        #
# ------------------------------------------------------------------ #

STATE_MENU      = "menu"
STATE_PLAYING   = "playing"
STATE_PAUSED    = "paused"
STATE_DONE      = "done"

# ------------------------------------------------------------------ #
#  HELPERS                                                           #
# ------------------------------------------------------------------ #

def draw_rounded_rect(surface, color, rect, radius=10, alpha=255):
    surf = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(surf, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
    surface.blit(surf, (rect[0], rect[1]))

def draw_text(surface, text, font, color, x, y, anchor="topleft"):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(**{anchor: (x, y)})
    surface.blit(rendered, rect)

def draw_bar(surface, x, y, w, h, value, max_value, color_fill, color_bg, radius=4):
    draw_rounded_rect(surface, color_bg, (x, y, w, h), radius)
    fill_w = int(w * min(value / max_value, 1.0))
    if fill_w > 0:
        draw_rounded_rect(surface, color_fill, (x, y, fill_w, h), radius)

# ------------------------------------------------------------------ #
#  CART-POLE RENDERER                                                #
# ------------------------------------------------------------------ #

def draw_cartpole(surface, state, sim_rect):
    cart_x_m, _, pole_angle, _ = state
    cx = sim_rect.centerx + int(cart_x_m * SCALE)
    cy = sim_rect.centery + 60

    # Ground track
    pygame.draw.rect(surface, C_TRACK,
                     (sim_rect.left + 20, cy + CART_H // 2 + WHEEL_R * 2 - 4,
                      sim_rect.width - 40, 6), border_radius=3)
 
    # Pole shadow
    pole_end_x = cx + int(math.sin(pole_angle) * POLE_LEN)
    pole_end_y = (cy - CART_H // 2) - int(math.cos(pole_angle) * POLE_LEN)
    pygame.draw.line(surface, (30, 35, 50),
                     (cx + 2, cy - CART_H // 2 + 2),
                     (pole_end_x + 2, pole_end_y + 2), POLE_W + 2)
 
    # Pole
    pole_color = C_POLE if abs(pole_angle) < 0.2 else C_RED
    pygame.draw.line(surface, pole_color,
                     (cx, cy - CART_H // 2),
                     (pole_end_x, pole_end_y), POLE_W)
    pygame.draw.circle(surface, pole_color, (pole_end_x, pole_end_y), POLE_W // 2 + 2)
 
    # Cart shadow
    draw_rounded_rect(surface, (10, 12, 20),
                      (cx - CART_W // 2 + 3, cy - CART_H // 2 + 3, CART_W, CART_H), radius=6)
 
    # Cart body
    draw_rounded_rect(surface, C_CART,
                      (cx - CART_W // 2, cy - CART_H // 2, CART_W, CART_H), radius=6)
 
    # Cart highlight
    draw_rounded_rect(surface, (150, 210, 255),
                      (cx - CART_W // 2 + 6, cy - CART_H // 2 + 4, CART_W - 12, 6),
                      radius=3, alpha=80)
 
    # Wheels
    for wx in [cx - CART_W // 2 + 16, cx + CART_W // 2 - 16]:
        wy = cy + CART_H // 2
        pygame.draw.circle(surface, C_CART_WHEEL, (wx, wy), WHEEL_R)
        pygame.draw.circle(surface, C_BG_SIM, (wx, wy), WHEEL_R - 4)
        pygame.draw.circle(surface, C_CART_WHEEL, (wx, wy), 3)
 
    # Center pin
    pygame.draw.circle(surface, C_ACCENT, (cx, cy - CART_H // 2), 5)

# ------------------------------------------------------------------ #
#  INFO PANEL                                                        #
# ------------------------------------------------------------------ #

def draw_panel(surface, fonts, stats, panel_rect, app_state):
    x = panel_rect.left + 20
    w = panel_rect.width - 40
    y = 30
 
    # Title
    draw_text(surface, "DQN", fonts["title"], C_ACCENT, panel_rect.centerx, y, anchor="midtop")
    y += 44
    draw_text(surface, "CartPole", fonts["subtitle"], C_TEXT_DIM, panel_rect.centerx, y, anchor="midtop")
    y += 40
 
    pygame.draw.line(surface, C_DIVIDER, (x, y), (x + w, y), 1)
    y += 20
 
    # Stats
    rows = [
        ("EPISODE", str(stats["episode"]),        C_ACCENT),
        ("STEP",    str(stats["step"]),            C_TEXT),
        ("REWARD",  f"{stats['reward']:.0f}",      C_GREEN),
        ("BEST",    f"{stats['best']:.0f}",        C_ACCENT2),
    ]
    for label, value, color in rows:
        draw_text(surface, label, fonts["label"], C_TEXT_DIM, x, y)
        draw_text(surface, value, fonts["value"], color, panel_rect.right - 20, y + 2, anchor="topright")
        y += 36
    y += 8
 
    pygame.draw.line(surface, C_DIVIDER, (x, y), (x + w, y), 1)
    y += 20
 
    # Reward bar
    draw_text(surface, "REWARD PROGRESS", fonts["label"], C_TEXT_DIM, x, y)
    y += 22
    draw_bar(surface, x, y, w, 10, stats["reward"], 500, C_GREEN, C_TRACK)
    y += 28
 
    # Pole angle bar
    pole_angle = abs(stats.get("pole_angle", 0))
    angle_color = C_RED if pole_angle > 0.15 else C_GREEN
    draw_text(surface, "POLE ANGLE", fonts["label"], C_TEXT_DIM, x, y)
    draw_text(surface, f"{math.degrees(pole_angle):.1f}deg",
              fonts["label"], angle_color, panel_rect.right - 20, y, anchor="topright")
    y += 22
    draw_bar(surface, x, y, w, 10, pole_angle, 0.418, angle_color, C_TRACK)
    y += 40
 
    pygame.draw.line(surface, C_DIVIDER, (x, y), (x + w, y), 1)
    y += 20
 
    # State values
    draw_text(surface, "STATE", fonts["label"], C_TEXT_DIM, x, y)
    y += 22
    labels = ["Cart Pos", "Cart Vel", "Pole Ang", "Pole Vel"]
    state  = stats.get("state", [0, 0, 0, 0])
    for lbl, val in zip(labels, state):
        draw_text(surface, lbl, fonts["tiny"], C_TEXT_DIM, x, y)
        draw_text(surface, f"{val:+.3f}", fonts["tiny"], C_TEXT,
                  panel_rect.right - 20, y, anchor="topright")
        y += 20
    y += 16
 
    pygame.draw.line(surface, C_DIVIDER, (x, y), (x + w, y), 1)
    y += 20
 
    # Last action
    draw_text(surface, "LAST ACTION", fonts["label"], C_TEXT_DIM, x, y)
    action_text  = "LEFT" if stats.get("action", 0) == 0 else "RIGHT"
    action_color = C_ACCENT2 if stats.get("action", 0) == 0 else C_ACCENT
    draw_text(surface, action_text, fonts["value"], action_color,
              panel_rect.right - 20, y + 2, anchor="topright")
    y += 50
 
    pygame.draw.line(surface, C_DIVIDER, (x, y), (x + w, y), 1)
    y += 16
 
    # Controls
    draw_text(surface, "CONTROLS", fonts["label"], C_TEXT_DIM, x, y)
    y += 22
    for key, desc in [("SPACE", "Play / Pause"), ("R", "Reset"), ("ESC", "Quit")]:
        draw_rounded_rect(surface, C_DIVIDER, (x, y - 1, 42, 20), radius=4)
        draw_text(surface, key, fonts["tiny"], C_ACCENT, x + 21, y + 9, anchor="center")
        draw_text(surface, desc, fonts["tiny"], C_TEXT_DIM, x + 50, y + 9, anchor="midleft")
        y += 26
 
    # Status badge
    status_map = {
        STATE_PLAYING : ("PLAYING",  C_GREEN),
        STATE_PAUSED  : ("PAUSED",   C_ACCENT2),
        STATE_DONE    : ("DONE",     C_RED),
        STATE_MENU    : ("MENU",     C_TEXT_DIM),
    }
    label, color = status_map.get(app_state, ("", C_TEXT_DIM))
    draw_rounded_rect(surface, C_TRACK, (x, WIN_H - 50, w, 30), radius=6)
    draw_text(surface, label, fonts["label"], color, panel_rect.centerx, WIN_H - 35, anchor="center")

# ------------------------------------------------------------------ #
#  MENU SCREEN                                                       #
# ------------------------------------------------------------------ #

def draw_menu(surface, fonts, model_found):
    surface.fill(C_BG_SIM)
    cx, cy = WIN_W // 2, WIN_H // 2
 
    draw_text(surface, "DQN CartPole", fonts["title_big"], C_ACCENT, cx, cy - 130, anchor="center")
    draw_text(surface, "Interactive Demo", fonts["subtitle"], C_TEXT_DIM, cx, cy - 78, anchor="center")
    pygame.draw.line(surface, C_DIVIDER, (cx - 150, cy - 50), (cx + 150, cy - 50), 1)
 
    if model_found:
        draw_text(surface, "Model loaded successfully", fonts["body"], C_GREEN, cx, cy - 24, anchor="center")
        draw_rounded_rect(surface, C_ACCENT, (cx - 100, cy + 20, 200, 50), radius=10)
        draw_text(surface, "PRESS SPACE TO PLAY", fonts["label"], C_BG_SIM, cx, cy + 45, anchor="center")
    else:
        draw_text(surface, "Model not found", fonts["body"], C_RED, cx, cy - 24, anchor="center")
        draw_text(surface, "Run train.py first, then relaunch this app.",
                  fonts["tiny"], C_TEXT_DIM, cx, cy + 10, anchor="center")
 
    draw_text(surface, "ESC to quit", fonts["tiny"], C_TEXT_DIM, cx, cy + 100, anchor="center")
 
 
# ------------------------------------------------------------------ #
#  MAIN                                                              #
# ------------------------------------------------------------------ #
 
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("DQN CartPole - Interactive Demo")
    clock = pygame.time.Clock()
 
    fonts = {
        "title_big" : pygame.font.SysFont("consolas", 42, bold=True),
        "title"     : pygame.font.SysFont("consolas", 28, bold=True),
        "subtitle"  : pygame.font.SysFont("consolas", 16),
        "value"     : pygame.font.SysFont("consolas", 20, bold=True),
        "label"     : pygame.font.SysFont("consolas", 13, bold=True),
        "body"      : pygame.font.SysFont("consolas", 15),
        "tiny"      : pygame.font.SysFont("consolas", 12),
    }
 
    sim_rect   = pygame.Rect(0, 0, SIM_W, SIM_H)
    panel_rect = pygame.Rect(SIM_W, 0, PANEL_W, WIN_H)
 
    model_found = os.path.exists(MODEL_PATH)
    agent = None
    if model_found:
        agent = DQNAgent(state_size=4, action_size=2, epsilon=0.0)
        agent.load(MODEL_PATH)
 
    env   = gym.make("CartPole-v1")
    state = np.zeros(4, dtype=np.float32)
 
    app_state = STATE_MENU
    stats = {
        "episode"    : 0,
        "step"       : 0,
        "reward"     : 0.0,
        "best"       : 0.0,
        "action"     : 0,
        "pole_angle" : 0.0,
        "state"      : [0.0, 0.0, 0.0, 0.0],
    }
 
    def reset_episode():
        nonlocal state
        obs, _ = env.reset()
        state  = np.array(obs, dtype=np.float32)
        stats["episode"] += 1
        stats["step"]     = 0
        stats["reward"]   = 0.0
        stats["state"]    = state.tolist()
 
    running = True
    while running:
 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
 
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    if app_state == STATE_MENU and model_found:
                        reset_episode()
                        app_state = STATE_PLAYING
                    elif app_state == STATE_PLAYING:
                        app_state = STATE_PAUSED
                    elif app_state == STATE_PAUSED:
                        app_state = STATE_PLAYING
                    elif app_state == STATE_DONE:
                        reset_episode()
                        app_state = STATE_PLAYING
                elif event.key == pygame.K_r:
                    if app_state in (STATE_PLAYING, STATE_PAUSED, STATE_DONE):
                        reset_episode()
                        app_state = STATE_PLAYING
 
        if app_state == STATE_PLAYING and agent is not None:
            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = np.array(next_state, dtype=np.float32)
 
            stats["step"]       += 1
            stats["reward"]     += float(reward)
            stats["action"]      = action
            stats["pole_angle"]  = float(state[2])
            stats["state"]       = state.tolist()
 
            if stats["reward"] > stats["best"]:
                stats["best"] = stats["reward"]
 
            if terminated or truncated:
                app_state = STATE_DONE
 
        if app_state == STATE_MENU:
            draw_menu(screen, fonts, model_found)
        else:
            pygame.draw.rect(screen, C_BG_SIM, sim_rect)
 
            # Subtle background grid
            for gx in range(0, SIM_W, 60):
                pygame.draw.line(screen, (22, 26, 38), (gx, 0), (gx, SIM_H))
            for gy in range(0, SIM_H, 60):
                pygame.draw.line(screen, (22, 26, 38), (0, gy), (SIM_W, gy))
 
            draw_cartpole(screen, state, sim_rect)
 
            if app_state == STATE_PAUSED:
                overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                screen.blit(overlay, (0, 0))
                draw_text(screen, "PAUSED", fonts["title_big"], C_ACCENT,
                          sim_rect.centerx, sim_rect.centery, anchor="center")
                draw_text(screen, "SPACE to resume", fonts["body"], C_TEXT_DIM,
                          sim_rect.centerx, sim_rect.centery + 50, anchor="center")
 
            if app_state == STATE_DONE:
                overlay = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 140))
                screen.blit(overlay, (0, 0))
                draw_text(screen, "EPISODE DONE", fonts["title"], C_ACCENT2,
                          sim_rect.centerx, sim_rect.centery - 30, anchor="center")
                draw_text(screen, f"Reward: {stats['reward']:.0f}", fonts["value"], C_TEXT,
                          sim_rect.centerx, sim_rect.centery + 10, anchor="center")
                draw_text(screen, "SPACE to restart  |  R to reset", fonts["body"], C_TEXT_DIM,
                          sim_rect.centerx, sim_rect.centery + 50, anchor="center")
 
            pygame.draw.rect(screen, C_BG_PANEL, panel_rect)
            pygame.draw.line(screen, C_DIVIDER, (SIM_W, 0), (SIM_W, WIN_H), 2)
            draw_panel(screen, fonts, stats, panel_rect, app_state)
 
        pygame.display.flip()
        clock.tick(FPS)
 
    env.close()
    pygame.quit()
    sys.exit()
 
 
if __name__ == "__main__":
    main()