import pygame
from game import gobang
pygame.init()

screen = pygame.display.set_mode([500, 500])
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:

            running = False
    screen.fill((255,255, 255))
    pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
    pygame.display.flip()
pygame.quit()
class gobangGUI:
    def __init__(self, args):
        self.args = args
        self.boardsize = args.boardsize
        self.start_x = 30
        self.start_y = 50
        self.edgesize = 30
    def draw(self, screen):
        pass
