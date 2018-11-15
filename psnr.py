
import numpy as np
a = 0
b = 1
mse = np.mean(np.square(a - b))
psnr_ = 10. * np.log10((255. * 255.) / mse)