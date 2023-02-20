import numpy as np
import pygame
import math
import random


def get_direction(randomness):
    """Return ball direction when ball is reset to the center. The default directions are 45, 135, 225 and 215 degrees, with vertical and horizontal deviations.

    Args:
        randomness (_type_): a value between 0 and 1 which sets the amount of deviation from the default directions. A higher value corresponds to more deviation.

    Returns:
        _type_: _description_
    """

    if randomness < 0 or randomness > 1:
        raise Exception(f"Parameter randomness should be between 0 and 1. You gave {randomness}.")
    max_vertical_deviation = 45
    max_horizontal_deviation = 45
    min_vertical_deviation = 25
    min_horizontal_deviation = 10

    vertical_deviation = max_vertical_deviation - randomness * (
        max_vertical_deviation - min_vertical_deviation
    )
    horizontal_deviation = max_horizontal_deviation - randomness * (
        max_horizontal_deviation - min_horizontal_deviation
    )

    valid_ranges = [
        (horizontal_deviation, 90 - vertical_deviation),
        (90 + vertical_deviation, 180 - horizontal_deviation),
        (180 + horizontal_deviation, 270 - vertical_deviation),
        (270 + vertical_deviation, 360 - horizontal_deviation),
    ]

    valid_ranges = [(math.radians(r[0]), math.radians(r[1])) for r in valid_ranges]

    angle = random.uniform(*random.choice(valid_ranges))

    return math.cos(angle), math.sin(angle)


def get_small_random_value(randomizer):
    return (1 / 100) * randomizer.random()


class Ball(pygame.sprite.Sprite):
    def __init__(
        self, randomizer, dims, speed, bounce_randomness, ball_direction_randomness
    ):
        self.surf = pygame.Surface(dims)
        self.rect = self.surf.get_rect()
        self.speed_val = speed

        direction = get_direction(ball_direction_randomness)
        self.speed = [
            int(self.speed_val * direction[0]),
            int(self.speed_val * direction[1]),
        ]
        self.bounce_randomness = bounce_randomness
        self.ball_direction_randomness = ball_direction_randomness
        self.done = False
        self.hit = False
        self.randomizer = randomizer
        # self.paddle_hits_0 = 0

    def reset(self, location):
        """Moves the ball to a new location and sets it's direction to a new random direction.

        Args:
            location (_type_): new ball location
        """
        self.rect.center = location

        direction = get_direction(self.ball_direction_randomness)

        self.speed = [
            int(self.speed_val * direction[0]),
            int(self.speed_val * direction[1]),
        ]

    def update2(self, area, p0, p1):
        """Moves the ball one step to its current direction. Handles ball collision to walls and paddles.
        Args:
            area (_type_): game area
            p0 (_type_): left paddle
            p1 (_type_): right paddle

        Returns:
            - termination (bool): True if ball hits left or right wall, False otherwise
            - wall (str): Either top, bottom, left or right if ball hits a wall, None otherwise
            - paddle_hit (bool): True if ball hits either paddle, False otherwise
        """
        self.rect.x += self.speed[0]
        self.rect.y += self.speed[1]

        terminated = False
        wall = None
        paddle_hit = False

        # ball hits a wall
        if not area.contains(self.rect):
            if self.rect.bottom > area.bottom:
                wall = "bottom"
                self.rect.bottom = area.bottom
                self.speed[1] = -self.speed[1]
            elif self.rect.top < area.top:
                wall = "top"
                self.rect.top = area.top
                self.speed[1] = -self.speed[1]
            else:
                if self.rect.left < area.left:
                    wall = "left"
                    terminated = True
                    return terminated, wall, paddle_hit
                elif self.rect.right > area.right:
                    wall = "right"
                    terminated = True
                    return terminated, wall, paddle_hit
                self.speed[0] = -self.speed[0]

        # ball hits a paddle
        else:
            r_val = 0
            if self.bounce_randomness:
                r_val = get_small_random_value(self.randomizer)

            # ball in left half of screen
            if self.rect.center[0] < area.center[0]:
                is_collision, self.rect, self.speed = p0.process_collision(
                    self.rect, self.speed, p0.location
                )
                if is_collision:
                    print("hit left paddle")
                    self.speed = [
                        self.speed[0] + np.sign(self.speed[0]) * r_val,
                        self.speed[1] + np.sign(self.speed[1]) * r_val,
                    ]
                    paddle_hit = True

            # ball in right half
            else:
                is_collision, self.rect, self.speed = p1.process_collision(
                    self.rect, self.speed, p1.location
                )
                if is_collision:
                    print("hit right paddle")
                    self.speed = [
                        self.speed[0] + np.sign(self.speed[0]) * r_val,
                        self.speed[1] + np.sign(self.speed[1]) * r_val,
                    ]
                    paddle_hit = True

        return terminated, wall, paddle_hit

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)
