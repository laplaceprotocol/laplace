import math
import numpy as np

WINDOWS = (1, 24, 168, 720)
HOURS_PER_YEAR = 8760
RELEASES = ("fomc", "cpi", "nfp")
FLOOR = 0.001  # final forecasts mix in the uniform distribution at this weight
# EV-HAR, exactly as in its own program: per asset and horizon its intercept, one coefficient per window of 5-minute
# realized variance and one for the DVOL-implied variance (COEFS); its overlay's coefficients (BETA); its tails (NU).
# Then one small network per horizon that corrects EV-HAR's log-probabilities. NET holds each network's tensors, in the
# order of SHAPES, as 16-bit two's-complement integers in hex, each tensor scaled by its own factor. All fitted on
# evolve-senses1's search window (2025-02-01 to 2026-06-01), stopped on its validation window (2026-06-01 to 2026-08-16).
COEFS = {
    'BTC': {
        1: (-1.2183486454517215e-05, 0.42588618636956244, 0.23392047823524964, 0.04998355550540318, -0.36221459343495643, 1.1408419876627751),
        4: (-5.2286976456116505e-05, 0.2019999753102948, 0.24389866032724652, 0.2004695502633158, -0.46460343056670783, 1.326481485316489),
        24: (-0.00034193711290370646, 0.05662199757717867, 0.16214153208845017, 0.3076295044201385, -0.545567491787376, 1.5763176035148476),
    },
    'ETH': {
        1: (-9.375270348138303e-06, 0.10938355384133946, 0.39496427685023905, 0.18629535244346224, -0.06141188874660297, 0.5737006370692128),
        4: (-3.5646294945870425e-05, 0.06160299379385614, 0.31176057278021624, 0.18482498800445352, -0.020530327000257115, 0.6320255202156564),
        24: (-9.289639791777574e-05, 0.021877565254894038, 0.24080872620080168, 0.19625976101243806, 0.003955057154652775, 0.6159942027525824),
    },
    'SOL': {
        1: (-1.5218100856269051e-05, 0.22028453776385148, 0.20197450379757304, 0.3237005120898702, 0.080629922373679, 1.0936766776481945),
        4: (-3.8372008475432654e-05, 0.12149984429685201, 0.2661264814093093, 0.2987350642636625, 0.03657841539940418, 1.1203168136799395),
        24: (-4.842658770808187e-05, 0.0436893027776904, 0.1817936705968305, 0.36856146899814807, 0.08511834028525725, 0.9263060925744995),
    },
    'XRP': {
        1: (-6.932927825844821e-05, 0.1295541771559212, 0.10917044721469549, 0.08969028381081293, 0.207375186711334, 4.057483117590346),
        4: (-0.00020153623250853076, 0.08548371908869468, 0.08954052442769933, 0.0744538216862454, 0.17370590471313857, 3.487126603090374),
        24: (-0.0010692506431404353, 0.029340687540072907, 0.060690943160908783, 0.05741428397336967, 0.223641895375727, 3.408833296293574),
    },
    'DOGE': {
        1: (-1.3682965220516839e-05, 0.28127804095406683, 0.2658568297620059, 0.11130534481562795, 0.09332604008883305, 1.2997305040619864),
        4: (-1.6550979624199904e-05, 0.1639921957401122, 0.24337983262671342, 0.11274798974423435, 0.1786252053424771, 1.0194244568717876),
        24: (6.850816472934005e-05, 0.06571525921274934, 0.17034481950050734, 0.1187294170049038, 0.24734307763476795, 1.0685849923544666),
    },
    'AVAX': {
        1: (-1.8043129498261188e-05, 0.3022448527766649, 0.01754747055367656, 0.06566821311057627, 0.09869012829222512, 2.423771277744707),
        4: (2.220263246589483e-05, 0.321271570544113, 0.05073576453301426, 0.05286512149451761, 0.10688781478374895, 1.324188138713839),
        24: (0.00024037375303006148, 0.11224373220379708, 0.030851690190729225, 0.06429691642289995, 0.15081578914906252, 1.7198321853405765),
    },
    'LINK': {
        1: (-2.643173840960433e-05, 0.048064594043678355, 0.27674226983763106, 0.25429208853012497, 0.1403962456258366, 1.7940103088659254),
        4: (-1.59751112323369e-05, 0.0532762163552124, 0.21402472179706503, 0.20409519320609293, 0.15186239517852643, 1.1791951436330064),
        24: (1.7043165899671052e-06, 0.022016530668735323, 0.10525824056599589, 0.22666418325738075, 0.2057017094389209, 1.2382604518698417),
    },
}
BETA = {
    1: (-0.3997644827906448, 0.11654877340272313, 0.17774983010113451, 0.03155690275576085, -0.10893903673077632, -0.24710988412282664, -0.1368834192230748, -0.15269558368033764, -0.15789259023247537, -0.09579391109442936, -0.09302889582269645, -0.2796519196727146, -0.25861462790108164, -0.09210195042344413, 0.3374020233945855, 0.625564193795262, 0.3695124525343765, 0.15258141685482957, 0.24581230965248593, -0.04509059581132868, -0.06283013090251334, 0.0013983270181413211, -0.10851441389251577, 0.08110230389717753, -0.29965411875892206, 0.8736922954235536, 0.1080470597192402, -0.3638692655707468, 0.2799229743665443, 0.271229876573757),
    4: (-0.46021540552950124, 0.12468240245380992, -0.08593729009302452, -0.19445304491121146, -0.19609173412172543, -0.21471311349471195, -0.10863151913487369, -0.08670776503843644, -0.1672606832787326, -0.18664828602395878, -0.18968412226032846, -0.10676736832960765, 0.0980593424983334, 0.3186708310551726, 0.4358280391457073, 0.41632662305643264, 0.29657578054302214, 0.09835618537353283, -0.01903221913915705, -0.17584636322565092, -0.03920802951835077, -0.06528381083708056, -0.03293525853150003, 0.03081305326886642, 0.049891304668033674, 0.00012591478671860137, 0.19319427644845966, -0.36051641900911013, 0.3395233166201064, 0.20457809873136334),
    24: (-0.4088600526044081, -0.16530994077438593, 0.11229207214528898, -0.3034998503811039, 0.34281497243200915, -0.4449512620291336),
}
NU = {1: 4.0, 4: 4.0, 24: 5.0}
SHAPES = ((27,), (27,), (27, 8), (8,), (8, 10), (10,))
NET = {
    1: ("7fffa94dc556e5d715ad90d8fef2fe14fbdb6cd19c85e8299ec7ff8d1e6e0000fffd0000fffdffb2ffee2d52008b01a1011604831d3433fe7767619b40ee309652b86af76a8b6a8d4d8f7fff44267f136ac060734b8f4b8d4b8f4b8d4b9f4b7d305706420ace08d611ce287ff23e06b22a99fe5ff53af08372dc1419fcc2f86ce39f09d41a3cf62a0a0b04c6f68cf35f097fbfc7ec4927e8f97105aafb6e040ff7e904460500eeb3fa0dffab052317b1cdeb1f9f179df313ffe8f447fe2a0ca427ff0f9601ebf85045ce18bb1dfe1a69e52001d5db98cb98fd33f35f0fdc0909f40e0478f0a90679f85310c41853128edc50081f0854ce0e0513eeebf26bf843169c00af0f0e1682e87907000910fc8dff39f49aacb30ac6e6921bc1f550ebc718c4f2a9ec6df70afac8efee014a058c179cf2acefae3873fcec2a1401fc03fcf9abfc2c15310c9609c4141ffc8a016cf907fc8301ce03d804a3f9b1ccafcfa6041fe03712bcec39f8702a790e690eeff9fb0acc1758271c05f503cf1056fef9e3572072015e080900d90ce40c7afcfff3fee9a50616ffaa129e0d8411a00f1a02a1f818e4c1d356fd7fff9a06c60c580853d212ee9c12020c05f7c5f056a3102c012308d46405d6fb1c36a5f62c119ef8e3e5d5f789e2bbf630f5b0fac13012051c26480d3103fad83008ebef43e5fae5de1c390ac319f610a20f40f4b2cdf5b78b361f10f2e852e824f01f5365e811f5c47fff1f897145dd4ee7ccd78bde0c276cd469e901eb697fffba93e0a5420938e91308e9a4be92d343e8cc0e0f1f0cdfca9e88ddd9199b4606559f342602ebd3dde3a60ea2dee3f42b1868413c45971ecceaa1bd34b873bf77b959dc0523715ea06e15375f0044cb1ab847522af8740d481b3e212420811667f3cdc1028001642c1194ef65cae9b87fcfa9f5a6148d1edb1ebbd1b294efd1d71fce7a2e7b8b2950ddcab742f3a5db7ac5aff074204747172e9b0957f670dcc5fbde0c1c5f3353f620f6ad6696a7f0143f422c608001", (2.4748353668779495e-05, 3.655813164062671e-05, 3.7685415644146735e-05, 6.079958900016544e-05, 3.11234448215053e-05, 1.8608063686843466e-05)),
    4: ("7fffac0ec733e6ac14fd9460fefbfe24fbfd695c9faee8eba1ddff911d760000fffd0000fffdffb4ffef2be1021b0194010d11791c462e3b7767619b40ee309652b86af76a8b6a8d4d8f7fff44267f136ac060734b8f4b8d4b8f4b8d4b9f4b7d2f750c740ace08d6220023f606b6f8f3222e0599e369134c4599328cda71bc77debc023583e9f453bd81d8ebe2f4ef1e15f837a4f92448eedc7f12c330380504d852cecee7920bc1ca1ef4d6f7811cde0b7512f7dd1bf91529d74fb5bccd50cd47d529b30e1def3f4dd1689f602846301313bf83ffc32309bb3d151716f12526c0cddcafe8b5c1915cd5e588118ff2335483467821cd8a6502c7247d3595df06613eb987dcb231e8029e6adf0a10de35ebd9c5c4bbcf21a51f044518d7ba5a26f0a95407e8e7d8bf0d55370d0898e0c0b86bb5331ad434d2f73bc6a92e98bbaad9f1212587e8149fe0f505d5fbf60ac9250cd7864bd726c12d083e2c2a092d0d0409b3daf8cbd6eeb4110dcbe1a8c90ac8560e5313254695ff42379b4ad73d6381c341931220169e13d0146e110cecd89a82fa30d51b39a6f0970475d2a8e4bfff62b0b9100cfa57c0b880011b62f01ee3b7f36560f2fd58049fb0a257ec30f838f2028b30ec13cc20baf7e1b15636e998bbf332f516c85720b9f54fe8af1bbfe6914a2212c2e6af983cba343a01fd4df0e457b4bb8be0bee6c71f4bdc96df43aa5526ed2b06e69308b8bb03e6d2498b073e7b4ed5fd1ef8f5911e3300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", (2.556047567288056e-05, 3.655813164062671e-05, 1.4085516835198125e-05, 1.0, 1.0, 1.0)),
    24: ("7fffb06aca26e7fd13e599f8ff08fe3dfc3263e3a4afea1ea6c1ff961bee0000fffd0000fffdffb8fff0299a0bfa017f00ff63641ace270a7767619b40ee309652b86af76a8b6a8d4d8f7fff44267f136ac060734b8f4b8d4b8f4b8d4b9f4b7d2a5f1d6c0ace08d631970d0006b6f8f3222e0599e369134c4599328cda71bc77debc023583e9f453bd81d8ebe2f4ef1e15f837a4f92448eedc7f12c330380504d852cecee7920bc1ca1ef4d6f7811cde0b7512f7dd1bf91529d74fb5bccd50cd47d529b30e1def3f4dd1689f602846301313bf83ffc32309bb3d151716f12526c0cddcafe8b5c1915cd5e588118ff2335483467821cd8a6502c7247d3595df06613eb987dcb231e8029e6adf0a10de35ebd9c5c4bbcf21a51f044518d7ba5a26f0a95407e8e7d8bf0d55370d0898e0c0b86bb5331ad434d2f73bc6a92e98bbaad9f1212587e8149fe0f505d5fbf60ac9250cd7864bd726c12d083e2c2a092d0d0409b3daf8cbd6eeb4110dcbe1a8c90ac8560e5313254695ff42379b4ad73d6381c341931220169e13d0146e110cecd89a82fa30d51b39a6f0970475d2a8e4bfff62b0b9100cfa57c0b880011b62f01ee3b7f36560f2fd58049fb0a257ec30f838f2028b30ec13cc20baf7e1b15636e998bbf332f516c85720b9f54fe8af1bbfe6914a2212c2e6af983cba343a01fd4df0e457b4bb8be0bee6c71f4bdc96df43aa5526ed2b06e69308b8bb03e6d2498b073e7b4ed5fd1ef8f5911e3300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", (2.6961021889730186e-05, 3.655813164062671e-05, 1.4085516835198125e-05, 1.0, 1.0, 1.0)),
}


def _filled_of(close, rv):
    """Hourly realized variance from 5-minute returns over a window, the squared hourly return where it is unknown."""
    r = np.diff(np.log(close))
    if rv is None:
        return r * r
    rv = rv[1:]
    return np.where(np.isfinite(rv), rv, r * r)


def _filled(ctx):
    return _filled_of(ctx.close, ctx.senses.get("rv"))


def _implied(ctx):
    """The hourly variance implied by the asset's DVOL index at the forecast hour, else by BTC's (NaN if neither)."""
    value = math.nan
    own = ctx.senses.get("dvol")
    if own is not None:
        value = float(own[-1])
    if not math.isfinite(value):
        market = ctx.peer_senses.get("BTC", {}).get("dvol")
        value = float(market[-1]) if market is not None else math.nan
    return (value / 100) ** 2 / HOURS_PER_YEAR


def _session_hours(events, horizon):
    """Hours of NYSE regular session inside the forecast's window."""
    total, opened = 0.0, None
    for hours, kind in events:
        if kind == "us-open":
            opened = hours
        elif kind == "us-close" and opened is not None:
            total += max(0.0, min(hours, horizon) - max(opened, 0.0))
            opened = None
    return total


def _inputs(ctx):
    """EV-HAR's overlay inputs: the window's weekend share, the UTC hour (1- and 4-hour horizons), then the scheduled
    releases inside, just before and just after the window, NYSE opens inside it, and its share of the session."""
    h = ctx.horizon
    x = [sum(1 for k in range(h) if (ctx.weekday + (ctx.hour + k) // 24) % 7 >= 5) / h]
    if h <= 4:
        x += [(1.0 if ctx.hour == j else 0.0) - 1 / 24 for j in range(24)]
    releases = [t for t, kind in ctx.events if kind in RELEASES]
    x.append(sum(1 for t in releases if 0 <= t < h))
    x.append(sum(1 for t in releases if -3 <= t < 0))
    x.append(sum(1 for t in releases if h <= t < h + 2))
    x.append(sum(1 for t, kind in ctx.events if kind == "us-open" and 0 <= t < h))
    x.append(_session_hours(ctx.events, h) / h)
    return x


def _tensors(h):
    """Horizon h's network: its tensors, decoded from 16-bit integers and scaled."""
    text, scales = NET[h]
    ints = [int(text[i : i + 4], 16) for i in range(0, len(text), 4)]
    ints = [v - 65536 if v >= 32768 else v for v in ints]
    out, i = [], 0
    for shape, scale in zip(SHAPES, scales):
        n = 1
        for d in shape:
            n *= d
        out.append(np.array(ints[i : i + n], dtype=np.float64).reshape(shape) * scale)
        i += n
    return out


TENSORS = {h: _tensors(h) for h in NET}


def _net_inputs(ctx, sd):
    """The network's inputs (laplace_research.evolve.learned.NAMES, in that order) from the window and EV-HAR's sd."""
    filled = _filled(ctx)
    week = max(float(np.mean(filled[-168:])), 1e-12)
    means = {w: max(float(np.mean(filled[-w:])), 1e-12) for w in (1, 4, 24, 720)}
    implied = _implied(ctx)
    if not math.isfinite(implied):
        implied = float(np.mean(filled[-720:]))
    close = ctx.close
    ret = {k: math.log(close[-1] / close[-1 - k]) for k in (1, 4, 24)}
    hl = np.log(ctx.high / ctx.low) ** 2 / (4 * math.log(2))
    btc = ctx.peers["BTC"]
    btc_filled = _filled_of(btc.close, ctx.peer_senses.get("BTC", {}).get("rv"))
    btc_week = max(float(np.mean(btc_filled[-168:])), 1e-12)
    funding = ctx.senses.get("funding")
    funding = float(funding[-1]) * 1e4 if funding is not None else 0.0
    overlay = _inputs(ctx)
    x = [
        math.log(max(sd, 1e-12) / float(ctx.edges[5])),
        *(math.log(means[k] / week) for k in (1, 4, 24, 720)),
        math.log(max(implied, 1e-12) / week),
        ret[1] / math.sqrt(week),
        ret[4] / math.sqrt(4 * week),
        ret[24] / math.sqrt(24 * week),
        abs(ret[1]) / math.sqrt(week),
        math.log(max(float(hl[-1]), 1e-12) / week),
        math.log(max(float(np.mean(hl[-24:])), 1e-12) / week),
        math.log(max(float(btc_filled[-1]), 1e-12) / btc_week),
        math.log(btc.close[-1] / btc.close[-2]) / math.sqrt(btc_week),
        funding,
        math.sin(2 * math.pi * ctx.hour / 24),
        math.cos(2 * math.pi * ctx.hour / 24),
        math.sin(2 * math.pi * ctx.hour / 12),
        math.cos(2 * math.pi * ctx.hour / 12),
        math.sin(2 * math.pi * ctx.weekday / 7),
        math.cos(2 * math.pi * ctx.weekday / 7),
        overlay[0],
        *overlay[-5:],
    ]
    return [v if math.isfinite(v) else 0.0 for v in x]


def forecast(ctx):
    """EV-NET: EV-HAR's distribution, corrected by a small network trained on the binned log score."""
    filled = _filled(ctx)
    means = [float(np.mean(filled[-w:])) for w in WINDOWS]
    implied = _implied(ctx)
    if not math.isfinite(implied):
        implied = means[3]
    c = COEFS[ctx.asset][ctx.horizon]
    var = c[0] + (sum(k * m for k, m in zip(c[1:5], means)) + c[5] * implied) * ctx.horizon
    floor = 0.1 * means[2] * ctx.horizon
    scale = math.exp(0.5 * sum(b * x for b, x in zip(BETA[ctx.horizon], _inputs(ctx))))
    sd = math.sqrt(max(var, floor)) * scale
    base = np.asarray(ctx.lib.student_t_probs(sd, NU[ctx.horizon], ctx.edges), dtype=np.float64)
    base = (1 - FLOOR) * base / base.sum() + FLOOR / len(base)
    mu, spread, w1, b1, w2, b2 = TENSORS[ctx.horizon]
    x = (np.array(_net_inputs(ctx, sd)) - mu) / spread
    logits = np.log(base) + np.tanh(x @ w1 + b1) @ w2 + b2
    p = np.exp(logits - logits.max())
    return p / p.sum()
