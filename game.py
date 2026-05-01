"""
game.py — Pygame Aim Trainer Game

Role:
- Build the main aim trainer game
- Includes adaptive difficulty system
"""

import pygame
import random
import time
import sys
from data_collector import DataCollector

# ── Window & Color Constants ─────────────────────────────────────────────
WINDOW_W, WINDOW_H = 800, 600

BLACK  = (15, 15, 20)
WHITE  = (240, 240, 240)
RED    = (220, 50, 50)
GREEN  = (50, 200, 100)
YELLOW = (240, 200, 40)
BLUE   = (60, 120, 220)
GRAY   = (80, 80, 90)
ORANGE = (240, 130, 40)

# ── Difficulty Settings ──────────────────────────────────────────────────
DIFFICULTY_SETTINGS = {
    1: {"target_size": 45, "speed": 0},  # Beginner
    2: {"target_size": 35, "speed": 1},  # Medium
    3: {"target_size": 25, "speed": 2},  # Hard
}

# ── Target Class ─────────────────────────────────────────────────────────
class Target:
    """A single clickable target on screen."""

    def __init__(self, difficulty: int):
        settings = DIFFICULTY_SETTINGS[difficulty]

        self.size = settings["target_size"]
        self.speed = settings["speed"]

        self.x = random.randint(self.size, WINDOW_W - self.size)
        self.y = random.randint(80, WINDOW_H - self.size)

        self.spawn_time = time.time()

        # Movement direction (for higher difficulty)
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

    def move(self):
        """Move target if difficulty allows."""
        if self.speed > 0:
            self.x += self.dx
            self.y += self.dy

            # Bounce off walls
            if self.x <= self.size or self.x >= WINDOW_W - self.size:
                self.dx *= -1
            if self.y <= 80 or self.y >= WINDOW_H - self.size:
                self.dy *= -1

    def draw(self, screen):
        """Draw target."""
        pygame.draw.circle(screen, RED, (self.x, self.y), self.size)
        pygame.draw.circle(screen, WHITE, (self.x, self.y), self.size - 8)
        pygame.draw.circle(screen, RED, (self.x, self.y), self.size - 16)
        pygame.draw.circle(screen, WHITE, (self.x, self.y), 5)

    def is_clicked(self, mouse_x: int, mouse_y: int) -> bool:
        """Check if target is clicked."""
        dist = ((mouse_x - self.x) ** 2 + (mouse_y - self.y) ** 2) ** 0.5
        return dist <= self.size

# ── Main Game Class ──────────────────────────────────────────────────────
class AimTrainer:
    """Main game class."""

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("AI Adaptive Aim Trainer")

        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_sm = pygame.font.SysFont("Arial", 20)

        # Game state
        self.hits = 0
        self.misses = 0
        self.difficulty = 1
        self.target = None
        self.running = True

        # Adaptive difficulty timing
        self.last_adjust_time = time.time()
        self.adjust_interval = 60

        # Data collector
        self.collector = DataCollector()

        self._spawn_target()

    # ── Core Helpers ─────────────────────────────────────────────────────
    def _spawn_target(self):
        self.target = Target(self.difficulty)

    def _accuracy(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 100.0
        return round((self.hits / total) * 100, 1)

    def _adjust_difficulty(self):
        acc = self._accuracy()

        if acc > 80 and self.difficulty < 3:
            self.difficulty += 1
            print(f"[Difficulty] Increased to {self.difficulty} (accuracy={acc}%)")
        elif acc < 60 and self.difficulty > 1:
            self.difficulty -= 1
            print(f"[Difficulty] Decreased to {self.difficulty} (accuracy={acc}%)")
        else:
            print(f"[Difficulty] Stays at {self.difficulty} (accuracy={acc}%)")

    # ── Drawing ──────────────────────────────────────────────────────────
    def _draw_hud(self):
        pygame.draw.rect(self.screen, (30, 30, 40), (0, 0, WINDOW_W, 60))
        pygame.draw.line(self.screen, GRAY, (0, 60), (WINDOW_W, 60), 1)

        diff_labels = {1: "Beginner", 2: "Medium", 3: "Hard"}
        diff_colors = {1: GREEN, 2: YELLOW, 3: ORANGE}

        stats = [
            (f"Hits: {self.hits}", GREEN, 20),
            (f"Misses: {self.misses}", RED, 180),
            (f"Accuracy: {self._accuracy()}%", WHITE, 340),
            (f"Level: {diff_labels[self.difficulty]}", diff_colors[self.difficulty], 520),
        ]

        for text, color, x in stats:
            surf = self.font_sm.render(text, True, color)
            self.screen.blit(surf, (x, 18))

        # Timer
        elapsed = time.time() - self.last_adjust_time
        remaining = max(0, int(self.adjust_interval - elapsed))

        timer_surf = self.font_sm.render(f"Next adjust: {remaining}s", True, GRAY)
        self.screen.blit(timer_surf, (WINDOW_W - 180, 18))

    # ── Main Loop ────────────────────────────────────────────────────────
    def run(self):
        while self.running:
            self.clock.tick(60)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = pygame.mouse.get_pos()

                    if my > 60:
                        self._handle_click(mx, my)

            # Difficulty update
            if time.time() - self.last_adjust_time >= self.adjust_interval:
                self._adjust_difficulty()
                self.last_adjust_time = time.time()

            # Update & Draw
            self.target.move()
            self.screen.fill(BLACK)

            self._draw_hud()
            self.target.draw(self.screen)

            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def _handle_click(self, mx: int, my: int):
        reaction_time = round(time.time() - self.target.spawn_time, 4)

        if self.target.is_clicked(mx, my):
            self.hits += 1
            hit = 1
        else:
            self.misses += 1
            hit = 0

        # Save data
        self.collector.record(
            reaction_time=reaction_time,
            target_x=self.target.x,
            target_y=self.target.y,
            target_size=self.target.size,
            hit=hit,
            difficulty=self.difficulty,
        )

        self._spawn_target()

# ── Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    game = AimTrainer()
    game.run()
