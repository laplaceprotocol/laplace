import math
import numpy as np


def forecast(ctx):
    """Seed: a zero-mean Laplace distribution scaled by the last week's realized volatility."""
    r = np.diff(np.log(ctx.close[-169:]))
    sd_hour = math.sqrt(float(np.mean(r * r)))
    return ctx.lib.laplace_probs(sd_hour * math.sqrt(ctx.horizon), ctx.edges)
