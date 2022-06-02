import numpy as np

from xrayvision.mem import resistant_mean


def test_resistant_mean():
    x = np.array([-0.61566182, -0.3362035, -0.73589487, -0.13014149, -0.91786147,
                  -0.02883366, 1.37688696, -0.22389746, -1.37010835, 0.21693273,
                  -0.82412723, 0.56856134, -0.36824209, -0.65785301, 0.10807291,
                  -0.16730037, -0.30308355, 0.12605051, -1.14017418, -1.36991364,
                  -0.45275646, 0.31636448, -0.1316372, -1.19194891, 0.58557167,
                  1.18596057, 0.01104923, 0.61225975, 0.133178, -0.37117463,
                  -0.14882964, -0.59864443, -0.49450362, 0.83710754, -0.51270022,
                  0.87380341, 0.52904637, 0.37973441, 0.34061863, -0.10493781,
                  0.86224542, -0.84283578, -0.35069229, -1.29952663, 0.39678443,
                  -0.94903521, 0.008999, 0.52126955, 0.24956007, -2.18242896,
                  0.15877853, -0.99771069, -0.14426402, 1.48957386, -0.68078661,
                  1.07328579, -2.57370777, 2.26526367, 1.20964125, -1.26854227,
                  0.20587741, -0.12066582, -0.78971368, -0.36412889, 0.12816666,
                  -0.50670234, 1.19987195, 1.29270178, 1.03906824, 1.41541003,
                  0.15639539, 0.92386204, -2.62109581, 0.68909503, -0.34981646,
                  0.93647975, -0.32268089, 1.67739013, -0.29216895, -2.30162528,
                  -0.89649313, 0.37481482, -2.18661087, 0.2190169, -0.30387412,
                  0.84001059, -0.25604226, 0.50463104, -1.25674997, 0.66314647,
                  -0.0073464, -0.59719724, 0.64527612, 1.25720287, 0.5032624,
                  0.07773715, 0.29573575, -0.80567372, -1.92934995, -2.0358119])
    rmean, rsigma = resistant_mean(x, 3)
    # values for IDL routine "RESISTANT_Mean, xx, 3, rmean, rsig, num"
    assert np.allclose(rmean, -0.10947954)
    assert np.allclose(rsigma, 0.096651256)
