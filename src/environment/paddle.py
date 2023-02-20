import pygame


class Paddle(pygame.sprite.Sprite):
    def __init__(self, dims, speed, location):
        self.surf = pygame.Surface(dims)
        self.rect = self.surf.get_rect()
        self.speed = speed
        self.location = location

    def reset(self, location, speed, seed=None):
        """Sets paddle to a new location, sets new speed.

        Args:
            location (_type_): new paddle location
            speed (_type_): new paddle speed
            seed (_type_, optional): random seed. Defaults to None.
        """
        if self.location == "left":
            self.rect.midleft = location
        elif self.location == "right":
            self.rect.midright = location

        self.speed = speed

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.rect)

    def update(self, area, action):
        """Moves paddle one step to its current direction. Does not move the paddle if the move would take out of game area.

        Args:
            area (_type_): game area
            action (_type_): movement, either 0: do not move, 1: up or 2: down.
        """
        move = [0, 0]
        if action > 0:
            if action == 1:
                move[1] = move[1] - self.speed
            elif action == 2:
                move[1] = move[1] + self.speed

            newpos = self.rect.move(move)
            if area.contains(newpos):
                self.rect = newpos

    def process_collision(self, ball_rect, ball_speed, paddle_location):
        """Handles a collision with ball. Returns the new location and speed for the ball if ball and paddle collide.

        Args:
            ball_rect : Ball rect
            ball_speed : Ball speed
            paddle_location: Paddle location

        Returns:
            is_collision: 1 if ball collides with paddle
            ball_rect: new ball rect
            ball_speed: new ball speed

        """
        if not self.rect.colliderect(ball_rect):
            return False, ball_rect, ball_speed
        # handle collision from left or right
        if paddle_location == "left" and ball_rect.left < self.rect.right:
            ball_rect.left = self.rect.right
            if ball_speed[0] < 0:
                ball_speed[0] *= -1
        elif paddle_location == "right" and ball_rect.right > self.rect.left:
            ball_rect.right = self.rect.left
            if ball_speed[0] > 0:
                ball_speed[0] *= -1
        # handle collision from top
        if (
            ball_rect.bottom > self.rect.top
            and ball_rect.top - ball_speed[1] < self.rect.top
            and ball_speed[1] > 0
        ):
            ball_rect.bottom = self.rect.top
            if ball_speed[1] > 0:
                ball_speed[1] *= -1
        # handle collision from bottom
        elif (
            ball_rect.top < self.rect.bottom
            and ball_rect.bottom - ball_speed[1] > self.rect.bottom
            and ball_speed[1] < 0
        ):
            ball_rect.top = self.rect.bottom - 1
            if ball_speed[1] < 0:
                ball_speed[1] *= -1
        return True, ball_rect, ball_speed
