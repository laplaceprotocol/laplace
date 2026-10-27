import math
import numpy as np

WINDOWS = (1, 24, 168, 720)
HOURS_PER_YEAR = 8760
RELEASES = ("fomc", "cpi", "nfp")
FLOOR = 0.001  # final forecasts mix in the uniform distribution at this weight
ASSETS = ('BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'AVAX', 'LINK')
# EV-HAR, exactly as in its own program: per asset and horizon its intercept, one coefficient per window of 5-minute
# realized variance and one for the DVOL-implied variance (COEFS); its overlay's coefficients (BETA); its tails (NU).
# Then boosted trees per horizon that correct EV-HAR's log-probabilities, in the program's parameter file: for horizon
# h, ctx.params holds "h<h>.root", ".feature", ".threshold", ".left", ".right" and ".value". All fitted on evolve-refit2610's panel from 2025-02-01 to 2026-09-27 00:00 UTC, the trees stopped on 2026-09-27 00:00 to 2026-10-27 00:00 UTC.
COEFS = {
    'BTC': {
        1: (-8.081929150817036e-06, 0.4231758258175121, 0.26684888078664676, -0.008075791211724692, -0.23892329909918847, 0.9124480702979931),
        4: (-3.97620982147441e-05, 0.1963708334487716, 0.28643383618051355, 0.1352142308235254, -0.41591649930551544, 1.21163387313233),
        24: (-0.0002483086326196499, 0.05610539134113579, 0.21411621854450413, 0.23102196988837528, -0.483508073874455, 1.4156804195820998),
    },
    'ETH': {
        1: (-1.9578712060613577e-06, 0.12113900987533666, 0.3992155302650561, 0.15525466606114674, -0.05907542890073372, 0.46312800147598177),
        4: (-2.1011662715683433e-05, 0.06409968438452818, 0.3216856973568902, 0.15726365855515217, -0.03741068748060153, 0.6060877392265815),
        24: (-5.9283716554566206e-05, 0.022129837538871933, 0.2518391069719359, 0.16547903571865302, -0.012769537963554067, 0.6327727434682335),
    },
    'SOL': {
        1: (-1.6626061861171218e-05, 0.23748882998832765, 0.20214428017276317, 0.2680737954931174, 0.06891499438776388, 1.3240847898290506),
        4: (-4.98726760678713e-05, 0.12288443657411163, 0.2756051783124641, 0.25574578150333976, 0.024385906584222766, 1.373839833639873),
        24: (-0.00015554607053535173, 0.043165451623022766, 0.19617119910394618, 0.3206562691747925, 0.06773748897383092, 1.2563272101846006),
    },
    'XRP': {
        1: (-5.8521301050096416e-05, 0.13230010926833957, 0.11570793833248672, 0.0898839628154578, 0.16495997684981972, 3.8638682070093107),
        4: (-0.00016042573025970322, 0.08721012904475811, 0.1013181941647221, 0.07371435790413655, 0.13170051740547503, 3.301874200480471),
        24: (-0.0008283028443258469, 0.03057810623880386, 0.07168335117556573, 0.055756646745753674, 0.18237429839862979, 3.227129257798491),
    },
    'DOGE': {
        1: (-1.2343787401797178e-05, 0.302042010934943, 0.25542270496258757, 0.08773217127586264, 0.06219509892479805, 1.4315580575844626),
        4: (-2.044258181108076e-05, 0.16612809939174494, 0.2486033438512251, 0.09575633847590878, 0.1610678151290956, 1.161865493438769),
        24: (-3.5924763165461524e-06, 0.06561102445824257, 0.18297728178941217, 0.09552855207850156, 0.23825455466638182, 1.2525721140946457),
    },
    'AVAX': {
        1: (-8.398320115367587e-06, 0.34168114278224604, 0.027733692519584685, 0.061770773447586885, 0.05625369584222789, 2.138985013891843),
        4: (1.7101464884977703e-05, 0.315759585765579, 0.06242746239508637, 0.05278028903981424, 0.08961814038214258, 1.4294482387174465),
        24: (0.00023581745909628316, 0.11907975662109571, 0.03993503922705072, 0.06510022183239914, 0.12604934110631405, 1.777112349779293),
    },
    'LINK': {
        1: (-2.0925467544312836e-05, 0.0506198767288073, 0.284434780932602, 0.24463677085905564, 0.09296972578201693, 1.8288654011147538),
        4: (-1.3778165856389674e-05, 0.0539851201157871, 0.2203438817731067, 0.19835513976537578, 0.13342790476183314, 1.241432030806355),
        24: (8.582295341085685e-07, 0.02214157542588735, 0.11271472620215658, 0.22030851728457032, 0.1902649120455061, 1.3027013516592516),
    },
}
BETA = {
    1: (-0.435130545008122, 0.07478777341041536, 0.2299581082081723, 0.040525532344812064, -0.12200634695572293, -0.2433973411039703, -0.1244165775851043, -0.2010749720471811, -0.1917002079688614, -0.01729838574590181, -0.1109144786556111, -0.30066957579331566, -0.26218008866383047, -0.01694894023606196, 0.30521214510324735, 0.6462113611464428, 0.3882521413462979, 0.1261922203450593, 0.25683518400081773, -0.063198215481762, -0.10173977310015987, -0.0587188472808445, -0.056833762998007145, 0.08985924452350821, -0.2864057542791151, 1.1815785918525925, 0.2055940361683799, -0.3891357315691127, 0.34109209234269655, 0.18069684291728788),
    4: (-0.4771036550490965, 0.10804905831484192, -0.05229728015170559, -0.2060032633252807, -0.20826349000942349, -0.24215434951291195, -0.09748912645943945, -0.10000479046161259, -0.18269986746941547, -0.21430778293607014, -0.19974457701643009, -0.12272887207445171, 0.12162988942411956, 0.35936956988557717, 0.43564697931914426, 0.46391449173737936, 0.303266259148427, 0.08321969060652988, -0.032505574472486365, -0.14257909715839298, -0.033899558021453884, -0.05293570137388691, -0.0650543164529974, 0.02729164090511575, 0.050420979091927755, 0.36105202822843074, 0.07516856477547185, -0.1983155201069071, 0.3669605579904928, 0.10368655315862295),
    24: (-0.4934281921144932, -0.14778331959583532, -0.020231295316756256, -0.25581721146165626, 0.36730877346550317, -0.5625271315142504),
}
NU = {1: 4.0, 4: 4.0, 24: 5.0}


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


def _walk(params, h, x):
    """Per bin, the sum of the leaves that inputs x reach in horizon h's trees. A node sends x left when
    x[feature] <= threshold; a child c >= 0 is a node, c < 0 is leaf ~c; tree t adds its leaf to bin t % 10."""
    feature, threshold = params[f"h{h}.feature"], params[f"h{h}.threshold"]
    left, right, value = params[f"h{h}.left"], params[f"h{h}.right"], params[f"h{h}.value"]
    node = np.array(params[f"h{h}.root"])
    inner = node >= 0
    while inner.any():
        k = np.where(inner, node, 0)
        node = np.where(inner, np.where(x[feature[k]] <= threshold[k], left[k], right[k]), node)
        inner = node >= 0
    return value[~node].reshape(-1, 10).sum(axis=0)


def _ev_har(ctx):
    """EV-HAR's standard deviation and its final distribution, as in its own program."""
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
    return sd, (1 - FLOOR) * base / base.sum() + FLOOR / len(base)


def forecast(ctx):
    """EV-GBM: EV-HAR's distribution, corrected by boosted trees trained on the binned log score."""
    sd, base = _ev_har(ctx)
    x = np.array([*_net_inputs(ctx, sd), ASSETS.index(ctx.asset), ctx.hour, ctx.weekday], dtype=np.float64)
    logits = np.log(base) + _walk(ctx.params, ctx.horizon, x)
    p = np.exp(logits - logits.max())
    return p / p.sum()
