import numpy as np
import cv2

colors = [(255, 66, 66), (66, 255, 66), (66, 66, 255), (66, 255, 255)]

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
            self.v += 0.165j - self.v / 54

class Block:
    def __init__(self, r):
        self.r = r
        self.v = -1j
        self.m = self.move()
        self.m.send(None)
    
    def move(self):
        while True:
            yield
            if self.r.imag <= 40:
                self.r += 666j
            self.r += self.v
        

v0 = 5.4 * np.random.randn(16).view(dtype=complex)
balls = [Ball(158+20j, v0[i]) for i in range(4)]
blocks = [Block(54 + i%2 * 54 + i//6 * 100 + i%6 * 111j) for i in range(18)]
data = [1] * 4
stack = [0] * 4
price = [1] * 4

class Bullet:
    def __init__(self, c, hp, r, v):
        self.cid = c
        self.hp = hp
        self.r = r
        self.v = v

def thgener(did, om=-0.0077):
    n0 = np.sqrt(2)/2 * [1+1j, -1+1j, 1-1j, -1-1j][did]
    eta = 1+0j
    while True:
        if eta.imag > np.sqrt(2)/2 and om > 0 or eta.imag < -np.sqrt(2)/2 and om < 0:
            om = - om
        eta *= np.exp(om * 1j)
        yield n0 * eta

canons = [thgener(0), thgener(1), thgener(2), thgener(3)]
poles = [336, 0] + np.array([[5, 5], [795, 5], [5, 795], [795, 795]], int)
mapcol = [(200, 66, 66), (66, 200, 66), (66, 66, 200), (66, 200, 200)]
bulcol = [(255, 99, 99), (66, 255, 66), (66, 66, 255), (66, 255, 255)]
board = np.hstack([np.vstack([(i*2+j)*np.ones((50, 50), int) for j in range(2)]) for i in range(2)])
bmap = np.array([[mapcol[board[i//8, j//8]] for i in range(800)] for j in range(800)], "uint8")
arr = np.vstack(20*[np.arange(-10, 10)])
ass = (np.dstack((arr.T, arr))**2).sum(2)
def getarea(hp):
    xp, yp = np.where(ass <= hp)
    return (xp-10, yp-10)
jmg = 100 * np.ones((800, 800, 3), dtype="uint8")
for i in range(4):
    cv2.circle(jmg, poles[i]-[336, 0], 50, mapcol[i], -1, cv2.LINE_AA)
img = np.hstack([33 * np.ones((800, 336, 3), dtype="uint8"), bmap])
bound = 180
ini = 1
tick = 0
isrunning = 600
bullets = []
loser = []

taker = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 60, (1136, 800))
while isrunning:
    iimg = img.copy()
    imps = np.zeros((26,), complex)
    for i in range(len(balls)):
        b = balls[i]
        if b is None:
            continue
        x, y = b.r.real, b.r.imag
        vx, vy = b.v.real, b.v.imag
        if x <= 10 and vx < 0 or x >= 323 and vx > 0:
            imps[i] += -2 * vx
        if y <= 20 and vy < 0:
            imps[i] += -2j * vy
        if y >= 646 and vy > 0:
            goal = data[i%4]
            if x > bound:
                if goal > 1000:
                    price[i%4] = 2
                stack[i%4] += goal
                data[i%4] = ini
            elif goal < 4000:
                data[i%4] += goal
            else:
                price[i%4] = 4
                stack[i%4] += 2 * goal
                data[i%4] = ini
            b.r = 158 + 20j
            b.v = 10 * np.random.randn()
        for j in range(i+1, len(balls)):
            c = balls[j]
            if c is None:
                continue
            dr = c.r - b.r
            dv = c.v - b.v
            if (dr * dr.conjugate()).real <= 400 and (dv * dr.conjugate()).real < 0:
                mom = dr * (dv * dr.conjugate()).real / (dr * dr.conjugate()).real
                imps[j] -= mom
                imps[i] += mom
        for c in blocks:
            dr = c.r - b.r
            dv = c.v - b.v
            if (dr * dr.conjugate()).real <= 900 and (dv * dr.conjugate()).real < 0:
                mom = dr * (dv * dr.conjugate()).real / (dr * dr.conjugate()).real
                imps[i] += 2 * mom
        b.m.send(imps[i])
        arg = np.rint(b.r)
        cv2.circle(iimg, (int(arg.real), int(arg.imag)), 10, colors[i%4], -1, cv2.LINE_AA)
    cv2.rectangle(iimg, (0, 646), (bound, 666), (66, 200, 66), -1)
    cv2.rectangle(iimg, (bound, 646), (336, 666), (66, 66, 200), -1)
    for i in range(4):
        cv2.putText(iimg, str(data[i]+stack[i]), (84*i+11, 721), cv2.FONT_HERSHEY_COMPLEX, 0.8, colors[i], 2, cv2.LINE_AA)
    for b in blocks:
        next(b.m)
        arg = np.rint(b.r)
        cv2.circle(iimg, (int(arg.real), int(arg.imag)), 20, (100, 100, 100), -1, cv2.LINE_AA)
    ths = [next(canons[i]) for i in range(4)]
    poplist = []
    for bid in range(len(bullets)):
        bul = bullets[bid]
        arg = np.rint(bul.r)
        x, y = int(arg.real), int(arg.imag)
        area = getarea(bul.hp)
        if (img[y+area[1], x+area[0]] != mapcol[bul.cid]).any():
            img[y+area[1], x+area[0]] = mapcol[bul.cid]
            bul.hp -= len(area[0])/20
            poplist.append(bid)
    while poplist:
        bid = poplist.pop()
        if bullets[bid].hp <= 5:
            bullets.pop(bid)
    for bul in bullets:
        bul.r += bul.v
        x, y = bul.r.real, bul.r.imag
        vx, vy = bul.v.real, bul.v.imag
        ra = int(np.sqrt(5*bul.hp))
        if x - ra <= 336 and vx < 0 or x + ra >= 1136 and vx > 0:
            bul.v -= 2*vx
        if y - ra <= 0 and vy < 0 or y + ra >= 800 and vy > 0:
            bul.v -= 2j*vy
        arg = np.rint(bul.r)
        cv2.circle(iimg, (int(arg.real), int(arg.imag)), int(np.sqrt(bul.hp)), bulcol[bul.cid], -1, cv2.LINE_AA)
    for i in range(4):
        if (img[poles[i][1]+np.arange(-4, 5), poles[i][0]+np.arange(-4, 5)] != mapcol[i]).any():
            data[i] = stack[i] = 0
            for j in range(len(balls)//4):
                balls[i + j*4] = None
            loser.append(i)
        if len(loser) == 3:
            isrunning -= 1
        cv2.line(iimg, tuple(poles[i]), np.rint(poles[i] + [50*ths[i].real, 50*ths[i].imag]).astype(int), (233, 233, 233), 2, cv2.LINE_AA)
        check = stack.copy(), data.copy()
        if stack[i] > 0:
            hp = min(price[i], stack[i])
            bullets.append(Bullet(i, 10*hp, poles[i][0] + 1j*poles[i][1], ths[i]*(1+0.002j)))
            bullets.append(Bullet(i, 10*hp, poles[i][0] + 1j*poles[i][1], ths[i]))
            bullets.append(Bullet(i, 10*hp, poles[i][0] + 1j*poles[i][1], ths[i]*(1-0.002j)))
            stack[i] -= hp
    cv2.imshow("test", iimg)
    cv2.waitKey(1)
    taker.write(iimg)
    if tick % 3600 == 3599:
        bound += 4
    if tick % 18000 == 17999:
        balls.extend([Ball(158+20j, v0[i]) for i in range(4)])
    tick += 1

taker.release()
