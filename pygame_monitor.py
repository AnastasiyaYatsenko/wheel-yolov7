import pygame
import GPUtil
from threading import Thread
import time
from contextlib import redirect_stdout
import sys, io

class PGMonitor(Thread):
    # display_surface = pygame.display.set_mode((400, 400))

    def __init__(self, delay):
        super(PGMonitor, self).__init__()
        self.stopped = False
        self.paused = False
        self.delay = delay  # Time between calls to GPUtil
        self.start()
        self.info = ""
        # pygame.init()
        X = 400
        Y = 400
        t = Thread(target=self.draw)
        t.start()
        # self.draw()

    def run(self):
        while not self.stopped:
            if not self.paused:
                f = io.StringIO()
                with redirect_stdout(f):
                    GPUtil.showUtilization()
                self.info = f.getvalue()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def draw(self):
        pygame.init()
        white = (255, 255, 255)
        green = (0, 255, 0)
        blue = (0, 0, 128)
        X = 400
        Y = 300
        self.display_surface = pygame.display.set_mode((X, Y))
        pygame.display.set_caption('GPUtil')
        font = pygame.font.Font('freesansbold.ttf', 40)
        while not self.stopped:
            myList = self.info.splitlines()
            self.display_surface.fill(blue)
            i = 0
            for line in myList:
                text = font.render(line, True, green, blue)
                textRect = text.get_rect()
                textRect.center = (X // 2, 25 + i*40)
                self.display_surface.blit(text, textRect)
                i += 1
            pygame.display.update()
        pygame.quit()