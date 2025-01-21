# This Python file uses the following encoding: utf-8

class Sector:
    ang = 6.67
    id = -1
    sector_ang = 5.0
    name = '0'
    center = (0, 0)
    corners = []

    def __init__(self, id, name):
        self.id = id
        self.name = name

    def set_coords(self, center, corners):
        self.center = center[:]
        self.corners = []
        for i in range(len(corners)):
            self.corners.append(corners[i])
        # print(f"in func: corners = {corners}; s.corners = {self.corners}")


class WheelTemplate:
    R_full = 100
    R_outer = 90
    R_inner = 80
    ang_rotate = 0
    center = (0, 0)
    wheel_nums = ['S', '1', '2', '5', '1', '10', '2', '5', '1', '8', '2',
                  'P', '1', '2', '1', '5', '1', '10', '1', '8', '1', '2',
                  'T', '1', '2', '5', '1', '2', '1', '8', '2', '1',
                  'P', '2', '1', '8', '1', '10', '1', '5', '1', '2', '1',
                  'T', '2', '8', '1', '5', '2', '10', '1', '5', '2', '1']
    # wheel_widths = []
    wheel_class_names = ['1', '10', '100', '2', '200', '300', '5', '8']
    wheel_class = {'1': 0,
                   '10': 1,
                   'T': 2,
                   '2': 3,
                   'S': 4,
                   'P': 5,
                   '5': 6,
                   '8': 7}

    def __init__(self):
        i = 0
        self.sectors = []
        for num in self.wheel_nums:
            s = Sector(i, num)
            self.sectors.append(s)
            i += 1

    def set_params(self, params):
        self.R_full = int(params[0])
        self.R_outer = int(params[1])
        self.R_inner = int(params[2])
        self.ang_rotate = float(params[3])
        self.center = (int(params[4]), int(params[5]))

    def getSectors(self):
        return self.sectors

    def rotate(self, ang):
        self.ang_rotate += ang

    def scale(self, add):
        self.R_full += add

    def scaleO(self, add):
        self.R_outer += add

    def scaleI(self, add):
        self.R_inner += add

    def move(self, stepX, stepY):
        new_x = self.center[0] + stepX
        new_y = self.center[1] + stepY
        self.center = (new_x, new_y)

    def calc_sectors(self):
        pass
