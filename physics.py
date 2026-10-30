"""
Mekanika orbitale per problemin Trappist Tour.
Vektoret jane tuple (x, y, z). Njesite: metra, sekonda.
"""
import math

AU = 149597870691.0          # njesia astronomike [m]
DAY = 86400.0                # nje dite [s]
EARTH_RADIUS = 6378137.0     # rrezja e Tokes [m]


# ---------- veprime me vektore ----------
def dot(a, b):
    """Prodhimi skalar a·b"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    """Prodhimi vektorial a×b"""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    """Gjatesia e vektorit |a|"""
    return math.sqrt(dot(a, a))


def sub(a, b):
    """a - b"""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, k):
    """a * k (k numer)"""
    return (a[0] * k, a[1] * k, a[2] * k)


def comb(a, ka, b, kb):
    """ka*a + kb*b"""
    return (ka * a[0] + kb * b[0], ka * a[1] + kb * b[1], ka * a[2] + kb * b[2])


# ---------- planetet ----------
class Planet:
    def __init__(self, t_ref, a, e, i, W, w, M0, mu_star, mu_self):
        # elementet e orbites: a gjysmeboshti, e ekscentriciteti, i pjerresia,
        # W nyja ngjitese, w argumenti i periapsisit, M0 anomalia mesatare ne t_ref
        self.t_ref, self.a, self.e, self.i, self.W, self.w, self.M0 = t_ref, a, e, i, W, w, M0
        self.mu_star, self.mu_self = mu_star, mu_self
        self.radius = EARTH_RADIUS
        self.n = math.sqrt(mu_star / a ** 3)            # levizja mesatare [rad/s]

    def eph(self, t_days):
        """Pozita dhe shpejtesia e planetit ne daten t (dite nga 2000-01-01)."""
        M = (self.M0 + self.n * (t_days - self.t_ref) * DAY) % (2 * math.pi)
        E = M
        for _ in range(30):                              # Newton: M = E - e sin E
            E -= (E - self.e * math.sin(E) - M) / (1 - self.e * math.cos(E))
        a, e = self.a, self.e
        b = a * math.sqrt(1 - e * e)
        x, y = a * (math.cos(E) - e), b * math.sin(E)    # pozita ne planin e orbites
        k = self.n / (1 - e * math.cos(E))
        vx, vy = -a * k * math.sin(E), b * k * math.cos(E)
        # rrotullimi nga plani i orbites ne 3D
        cW, sW, cw, sw = math.cos(self.W), math.sin(self.W), math.cos(self.w), math.sin(self.w)
        ci, si = math.cos(self.i), math.sin(self.i)
        R = ((cW * cw - sW * sw * ci, -cW * sw - sW * cw * ci),
             (sW * cw + cW * sw * ci, -sW * sw + cW * cw * ci),
             (sw * si, cw * si))
        return (tuple(R[m][0] * x + R[m][1] * y for m in range(3)),
                tuple(R[m][0] * vx + R[m][1] * vy for m in range(3)))