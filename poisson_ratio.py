import numpy as np

pi = np.pi
thetas = [pi/2., 5.*pi/12., pi/3., pi/4., pi/6., pi/12., 0]

for idx0 in range(len(thetas)):
    for idx1 in range(idx0+1, len(thetas)):
        t0, t1 = thetas[idx0], thetas[idx1]
        print t0, t1, "==>", (np.sin(t0) - np.sin(t1))/(np.cos(t1) - np.cos(t0))
    print "  "


