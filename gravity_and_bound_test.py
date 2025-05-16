import numpy as np
import cv2

colors = [(255, 66, 66), (66, 255, 66), (66, 66, 255), (66, 255, 255)] * 2 + [(100, 100, 100)] * 18

class Ball:
    def __init__(self, r, v):
        self.r = r
        self.v = v
        self.m = self.move()
        self.m.send(None)
        
    def move(self):
        while True:
            self.v += yield
            self.r += self.v
            self.v += 0.33j - self.v / 33

class Block:
    def __init__(self, r):
        self.r = r
        self.v = -1.5j
        self.m = self.move()
        self.m.send(None)
    
    def move(self):
        while True:
            yield
            if self.r.imag <= 40:
                self.r += 666j
            self.r += self.v
        

v0 = 10 * np.random.randn(16).view(dtype=complex)
balls = [Ball(158+20j, v0[i]) for i in range(8)] + [Block(22 + i%3 * 33 + i//6 * 111 + i % 6 * 111j) for i in range(18)]
data = [1] * 4
taker = cv2.VideoWriter('test.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 60, (336, 666))
while True:
    imps = np.zeros((26,), complex)
    for i in range(8):
        b = balls[i]
        x, y = b.r.real, b.r.imag
        vx, vy = b.v.real, b.v.imag
        if x <= 10 and vx < 0 or x >= 323 and vx > 0:
            imps[i] += -2 * vx
        if y <= 20 and vy < 0:
            imps[i] += -2j * vy
        if y >= 646 and vy > 0:
            goal = data[i%4]
            if x < 200:
                print(goal)
                data[i%4] = 1
            else:
                data[i%4] += goal
            b.r = 158 + 20j
            b.v = 10 * np.random.randn()
        for j in range(i+1, 8):
            c = balls[j]
            dr = c.r - b.r
            dv = c.v - b.v
            if (dr * dr.conjugate()).real <= 400 and (dv * dr.conjugate()).real < 0:
                mom = dr * (dv * dr.conjugate()).real / (dr * dr.conjugate()).real
                imps[j] -= mom
                imps[i] += mom
        for j in range(8, 26):
            c = balls[j]
            dr = c.r - b.r
            dv = c.v - b.v
            if (dr * dr.conjugate()).real <= 900 and (dv * dr.conjugate()).real < 0:
                mom = dr * (dv * dr.conjugate()).real / (dr * dr.conjugate()).real
                imps[i] += 2 * mom
    img = 33 * np.ones((666, 336, 3), dtype="uint8")
    cv2.rectangle(img, (0, 646), (200, 666), (66, 66, 200), -1)
    cv2.rectangle(img, (200, 646), (336, 666), (66, 200, 66), -1)
    for i in range(26):
        b = balls[i]
        b.m.send(imps[i])
        arg = np.rint(b.r)
        cv2.circle(img, (int(arg.real), int(arg.imag)), 10 if i < 8 else 20, colors[i], -1)
    cv2.imshow("test", img)
    taker.write(img)
    cv2.waitKey(8)
    if cv2.getWindowProperty('test', cv2.WND_PROP_VISIBLE) < 1:
        taker.release()
        break