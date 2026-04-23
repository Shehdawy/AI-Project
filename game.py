"""
game.py — Pygame Aim Trainer
Game Developer Module
Handles: target spawning, click detection, scoring, reaction time, difficulty scaling
"""

import pygame
import random
import time
import math
from data_collector import DataCollector

# ─── Constants ────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 600
FPS = 60
FONT_NAME = "monospace"

# Difficulty configs: (target_size, target_speed, spawn_interval_seconds)
DIFFICULTY_CONFIG = {
    "Beginner": {"size": 50, "speed": 1.5, "interval": 2.0},
    "Average":  {"size": 32, "speed": 2.5, "interval": 1.2},
    "Pro":      {"size": 18, "speed": 4.0, "interval": 0.7},
}

# Colors
BG_COLOR       = (15, 15, 25)
TARGET_COLOR   = (220, 60, 60)
TARGET_OUTLINE = (255, 120, 120)
HIT_COLOR      = (80, 220, 120)
MISS_COLOR     = (220, 80, 80)
TEXT_COLOR     = (220, 220, 240)
DIM_COLOR      = (100, 100, 120)
HUD_BG         = (25, 25, 40)


class Target:
    """A single circular target on screen."""

    def __init__(self, x, y, size, speed):
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        self.alive = True
        self.spawn_time = time.time()
        # Pulsing animation
        self.pulse = 0

    def update(self):
        self.pulse = (self.pulse + 0.08) % (2 * math.pi)

    def draw(self, surface):
        if not self.alive:
            return
        pulse_r = int(self.size + 4 * math.sin(self.pulse))
        # Glow ring
        pygame.draw.circle(surface, (80, 20, 20), (self.x, self.y), pulse_r + 6)
        # Main target
        pygame.draw.circle(surface, TARGET_COLOR, (self.x, self.y), pulse_r)
        # Inner ring
        pygame.draw.circle(surface, TARGET_OUTLINE, (self.x, self.y), max(pulse_r - 8, 4), 2)
        # Bullseye dot
        pygame.draw.circle(surface, (255, 220, 220), (self.x, self.y), max(pulse_r // 4, 3))

    def is_hit(self, mx, my):
        dist = math.hypot(mx - self.x, my - self.y)
        return dist <= self.size


class AimTrainerGame:
    """Main game class. Call run() to start."""

    def __init__(self, session_id="session_001"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🎯  AI Aim Trainer")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont(FONT_NAME, 28, bold=True)
        self.font_med   = pygame.font.SysFont(FONT_NAME, 20)
        self.font_small = pygame.font.SysFont(FONT_NAME, 15)

        self.collector = DataCollector(session_id)
        self.reset()

    # ── State ──────────────────────────────────────────────────────────────────
    def reset(self):
        self.targets        = []
        self.score          = 0
        self.misses         = 0
        self.total_shots    = 0
        self.reaction_times = []
        self.difficulty     = "Beginner"
        self.last_spawn     = time.time()
        self.game_over      = False
        self.running        = True
        self.flash_msg      = ""          # "HIT" / "MISS"
        self.flash_timer    = 0
        self.session_start  = time.time()
        self.ai_skill_label = "Beginner"  # updated externally by model

    # ── Difficulty ─────────────────────────────────────────────────────────────
    def set_difficulty(self, label: str):
        """Called externally by the ML model to update difficulty."""
        if label in DIFFICULTY_CONFIG:
            self.difficulty = label
            self.ai_skill_label = label

    def _cfg(self):
        return DIFFICULTY_CONFIG[self.difficulty]

    # ── Spawning ───────────────────────────────────────────────────────────────
    def _maybe_spawn(self):
        now = time.time()
        if now - self.last_spawn >= self._cfg()["interval"] and len(self.targets) < 3:
            margin = 60
            x = random.randint(margin, WIDTH - margin)
            y = random.randint(margin + 60, HEIGHT - margin)
            t = Target(x, y, self._cfg()["size"], self._cfg()["speed"])
            self.targets.append(t)
            self.last_spawn = now

    # ── Click handling ─────────────────────────────────────────────────────────
    def _handle_click(self, mx, my):
        self.total_shots += 1
        hit_something = False

        for t in self.targets:
            if t.alive and t.is_hit(mx, my):
                reaction = time.time() - t.spawn_time
                self.reaction_times.append(reaction)
                self.score += 1
                t.alive = False
                hit_something = True
                self.flash_msg   = "HIT"
                self.flash_timer = 30

                # Save to CSV
                self.collector.record(
                    reaction_time   = round(reaction, 4),
                    target_x        = t.x,
                    target_y        = t.y,
                    target_size     = t.size,
                    hit             = 1,
                    difficulty      = self.difficulty,
                )
                break

        if not hit_something:
            self.misses += 1
            self.flash_msg   = "MISS"
            self.flash_timer = 20
            self.collector.record(
                reaction_time = 0,
                target_x      = mx,
                target_y      = my,
                target_size   = 0,
                hit           = 0,
                difficulty    = self.difficulty,
            )

        # Remove dead targets
        self.targets = [t for t in self.targets if t.alive]

    # ── HUD ────────────────────────────────────────────────────────────────────
    def _draw_hud(self):
        # Top bar background
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, WIDTH, 55))
        pygame.draw.line(self.screen, (50, 50, 80), (0, 55), (WIDTH, 55), 1)

        accuracy = (self.score / self.total_shots * 100) if self.total_shots > 0 else 0
        avg_rt   = (sum(self.reaction_times) / len(self.reaction_times)) if self.reaction_times else 0
        elapsed  = int(time.time() - self.session_start)

        stats = [
            f"SCORE  {self.score}",
            f"ACC  {accuracy:.1f}%",
            f"AVG RT  {avg_rt:.3f}s",
            f"DIFF  {self.difficulty}",
            f"TIME  {elapsed}s",
            f"AI  {self.ai_skill_label}",
        ]
        x = 10
        for s in stats:
            surf = self.font_small.render(s, True, TEXT_COLOR)
            self.screen.blit(surf, (x, 18))
            x += 130

    def _draw_flash(self):
        if self.flash_timer > 0:
            color = HIT_COLOR if self.flash_msg == "HIT" else MISS_COLOR
            surf = self.font_large.render(self.flash_msg, True, color)
            self.screen.blit(surf, (WIDTH // 2 - surf.get_width() // 2, HEIGHT // 2 - 20))
            self.flash_timer -= 1

    def _draw_crosshair(self, mx, my):
        size, gap = 10, 4
        pygame.draw.line(self.screen, (200, 200, 220), (mx - size - gap, my), (mx - gap, my), 1)
        pygame.draw.line(self.screen, (200, 200, 220), (mx + gap, my), (mx + size + gap, my), 1)
        pygame.draw.line(self.screen, (200, 200, 220), (mx, my - size - gap), (mx, my - gap), 1)
        pygame.draw.line(self.screen, (200, 200, 220), (mx, my + gap), (mx, my + size + gap), 1)
        pygame.draw.circle(self.screen, (200, 200, 220), (mx, my), 2)

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self):
        pygame.mouse.set_visible(False)  # hide default cursor

        while self.running:
            self.clock.tick(FPS)
            mx, my = pygame.mouse.get_pos()

            # ── Events ─────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_r:
                        self.reset()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(mx, my)

            # ── Logic ──────────────────────────────────────────────────────────
            self._maybe_spawn()
            for t in self.targets:
                t.update()

            # ── Draw ───────────────────────────────────────────────────────────
            self.screen.fill(BG_COLOR)

            # Subtle grid background
            for gx in range(0, WIDTH, 40):
                pygame.draw.line(self.screen, (22, 22, 35), (gx, 55), (gx, HEIGHT), 1)
            for gy in range(60, HEIGHT, 40):
                pygame.draw.line(self.screen, (22, 22, 35), (0, gy), (WIDTH, gy), 1)

            for t in self.targets:
                t.draw(self.screen)

            self._draw_hud()
            self._draw_flash()
            self._draw_crosshair(mx, my)

            pygame.display.flip()

        pygame.mouse.set_visible(True)
        self.collector.save()
        pygame.quit()
        print(f"[Game] Session ended. Data saved. Score={self.score}")


if __name__ == "__main__":
    game = AimTrainerGame()
    game.run()
