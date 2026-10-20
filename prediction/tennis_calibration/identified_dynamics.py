"""Position-identifiable drag/Magnus dynamics and conditional impact laws.

The state is [p_m, v_m_s, b_eff_per_s].  ``b_eff`` is the identifiable
combination multiplying cross(b,v), not measured spin.  Contact spin conversion
is conditional on a fixed linear lift slope and thin-shell inertia assumption.
Training helpers are offline reconstruction; they are never causal forecasts.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from .flight import GRAVITY, initial_guess

ASSUMPTIONS = {
    'mass_kg': .0577, 'radius_m': .0335, 'air_density_kg_m3': 1.225,
    'magnus_slope': .6, 'inertia_factor': 2/3,
    'status': 'fixed assumptions; mass, radius, air density and true spin unmeasured',
}


def magnus_scale(parameters=None):
    """b_eff = k_M omega for C_L = magnus_slope * omega R / speed."""
    p = {**ASSUMPTIONS, **(parameters or {})}
    m, r, rho, slope = (float(p[k]) for k in
                        ('mass_kg', 'radius_m', 'air_density_kg_m3', 'magnus_slope'))
    if min(m, r, rho, slope) <= 0:
        raise ValueError('positive fixed ball/aerodynamic assumptions required')
    return .5 * rho * np.pi * r*r * slope * r / m


def drag_coefficient(k_drag, parameters=None):
    p = {**ASSUMPTIONS, **(parameters or {})}
    return 2 * p['mass_kg'] * float(k_drag) / (p['air_density_kg_m3'] * np.pi * p['radius_m']**2)


def skew(a):
    a = np.asarray(a, dtype=float)
    out = np.zeros(a.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -a[..., 2], a[..., 1]
    out[..., 1, 0], out[..., 1, 2] = a[..., 2], -a[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -a[..., 1], a[..., 0]
    return out


def _validate(states, k_drag):
    state = np.asarray(states, dtype=float)
    if state.shape[-1] != 9 or not np.isfinite(state).all():
        raise ValueError('finite 9D [p,v,b_eff] states required')
    if not np.isfinite(k_drag) or k_drag < 0:
        raise ValueError('finite nonnegative drag required')
    return state


def rhs(states, k_drag):
    state = _validate(states, k_drag)
    v, b = state[..., 3:6], state[..., 6:9]
    result = np.zeros_like(state)
    result[..., :3] = v
    result[..., 3:6] = GRAVITY - k_drag * np.linalg.norm(v, axis=-1, keepdims=True) * v + np.cross(b, v)
    return result


def rhs_jacobian(states, k_drag):
    state = _validate(states, k_drag)
    v, b = state[..., 3:6], state[..., 6:9]
    speed = np.linalg.norm(v, axis=-1)
    out = np.zeros(state.shape[:-1] + (9, 9))
    out[..., :3, 3:6] = np.eye(3)
    out[..., 3:6, 3:6] = (-k_drag * (speed[..., None, None] * np.eye(3)
        + v[..., :, None] * v[..., None, :] / np.maximum(speed[..., None, None], 1e-15)) + skew(b))
    out[..., 3:6, 6:9] = -skew(v)
    return out


def _rk4(states, h, k_drag, sensitivity=None):
    """One same-discretization RK4 step, optional 9x10 sensitivity."""
    h = np.asarray(h, dtype=float)
    while h.ndim < states.ndim:
        h = h[..., None]
    h_j = h[..., None]

    def stage(x, s):
        f = rhs(x, k_drag)
        if s is None:
            return f, None
        ds = rhs_jacobian(x, k_drag) @ s
        if s.shape[-1] == 10:
            v = x[..., 3:6]
            ds[..., 3:6, 9] -= np.linalg.norm(v, axis=-1, keepdims=True) * v
        return f, ds

    f1, j1 = stage(states, sensitivity)
    x2 = states + h*f1/2
    s2 = None if sensitivity is None else sensitivity + h_j*j1/2
    f2, j2 = stage(x2, s2)
    x3 = states + h*f2/2
    s3 = None if sensitivity is None else sensitivity + h_j*j2/2
    f3, j3 = stage(x3, s3)
    x4 = states + h*f3
    s4 = None if sensitivity is None else sensitivity + h_j*j3
    f4, j4 = stage(x4, s4)
    result = states + h*(f1 + 2*f2 + 2*f3 + f4)/6
    if sensitivity is None:
        return result, None
    return result, sensitivity + h_j*(j1 + 2*j2 + 2*j3 + j4)/6


def step(states, dt_s, k_drag, max_step_s=.004):
    state = _validate(states, k_drag).copy()
    if max_step_s <= 0 or not np.isfinite(dt_s).all():
        raise ValueError('finite duration and positive maximum step required')
    count = max(1, int(np.ceil(np.max(np.abs(dt_s)) / max_step_s)))
    for _ in range(count):
        state, _ = _rk4(state, np.asarray(dt_s)/count, k_drag)
    return state


def step_with_jacobian(states, dt_s, k_drag, max_step_s=.004):
    state = _validate(states, k_drag).copy()
    if max_step_s <= 0 or not np.isfinite(dt_s).all():
        raise ValueError('finite duration and positive maximum step required')
    sensitivity = np.broadcast_to(np.eye(9), state.shape[:-1] + (9, 9)).copy()
    count = max(1, int(np.ceil(np.max(np.abs(dt_s)) / max_step_s)))
    for _ in range(count):
        state, sensitivity = _rk4(state, np.asarray(dt_s)/count, k_drag, sensitivity)
    return state, sensitivity


def _times_and_states(times, initial_states, k_drag, max_step_s):
    times, states = np.asarray(times, dtype=float), _validate(initial_states, k_drag)
    if times.ndim != 2 or states.shape != (len(times), 9) or not times.size:
        raise ValueError('times NxM and initial_states Nx9 required')
    if np.any(times < 0) or np.any(np.diff(times, axis=1) < 0) or not np.isfinite(times).all() or max_step_s <= 0:
        raise ValueError('nonnegative monotonic times and positive step required')
    return times, states.copy()


def simulate_batch(times, initial_states, k_drag, max_step_s=.004):
    times, states = _times_and_states(times, initial_states, k_drag, max_step_s)
    out = np.empty(times.shape + (9,))
    previous = np.zeros(len(times))
    for j in range(times.shape[1]):
        states = step(states, times[:, j]-previous, k_drag, max_step_s)
        out[:, j] = states
        previous = times[:, j]
    return out


def simulate(times, initial_state, k_drag, max_step_s=.004):
    return simulate_batch(np.asarray(times)[None, :], np.asarray(initial_state)[None, :], k_drag, max_step_s)[0]


def simulate_batch_with_jacobian(times, initial_states, k_drag, max_step_s=.004):
    """Positions plus sensitivity to [p0(3),v0(3),b_eff(3),k_D]."""
    times, states = _times_and_states(times, initial_states, k_drag, max_step_s)
    sens = np.zeros((len(times), 9, 10))
    sens[:, :, :9] = np.eye(9)
    out, jac = np.empty(times.shape+(3,)), np.empty(times.shape+(3, 10))
    previous = np.zeros(len(times))
    for j in range(times.shape[1]):
        dt = times[:, j]-previous
        count = max(1, int(np.ceil(np.max(dt)/max_step_s)))
        for _ in range(count):
            states, sens = _rk4(states, dt/count, k_drag, sens)
        out[:, j], jac[:, j] = states[:, :3], sens[:, :3]
        previous = times[:, j]
    return out, jac


def impact_map(states, parameters, *, return_diagnostics=False):
    """Apply first ground impact to 9D state(s), under fixed model assumptions.

    Effective mode preserves b (an unvalidated postimpact aerodynamic prior).
    Stick/slip mode converts b to conditional spin with a fixed lift slope.
    Its friction parameter is conditional, not independently measured mu.
    """
    state = _validate(states, 0).copy()
    p = {**ASSUMPTIONS, **parameters}
    mode = p.get('impact_mode', 'effective')
    v = state[..., 3:6].copy()
    incoming = v[..., 2] < 0
    normal_speed = np.maximum(-v[..., 2], 0)
    e = np.clip(float(p.get('e_n', .834)) + float(p.get('normal_speed_slope', 0))
                * (normal_speed-float(p.get('normal_speed_reference_m_s', 5))), 0, 1)
    if mode == 'effective':
        alpha = float(p.get('alpha_t', .558))
        changed = v * np.stack((np.full_like(e, alpha), np.full_like(e, alpha), -e), axis=-1)
        state[..., 3:6] = np.where(incoming[..., None], changed, v)
        diagnostic = {'incoming': incoming, 'regime': np.where(incoming, 'effective', 'no_impact')}
    elif mode == 'stick_slip':
        mu = float(p.get('mu', .5))
        m, r = float(p['mass_kg']), float(p['radius_m'])
        inertia = float(p['inertia_factor']) * m*r*r
        if not 0 <= mu <= 1 or inertia <= 0:
            raise ValueError('conditional mu in [0,1] and positive inertia required')
        scale = magnus_scale(p)
        omega = state[..., 6:9]/scale
        lever = np.array([0., 0., -r])
        slip = v + np.cross(omega, lever)
        slip[..., 2] = 0
        j_normal = (1+e)*m*normal_speed
        j_stick = -slip/(1/m+r*r/inertia)
        stick_norm = np.linalg.norm(j_stick, axis=-1)
        slip_limit = mu*j_normal
        fraction = np.minimum(1, slip_limit/np.maximum(stick_norm, 1e-15))
        j_tangent = j_stick*fraction[..., None]
        j_tangent = np.where(incoming[..., None], j_tangent, 0)
        impulse = j_tangent.copy()
        impulse[..., 2] = j_normal
        state[..., 3:6] = v+impulse/m
        state[..., 6:9] = scale*(omega+np.cross(lever, impulse)/inertia)
        diagnostic = {'incoming': incoming, 'impulse_N_s': impulse,
            'regime': np.where(~incoming, 'no_impact', np.where(stick_norm <= slip_limit, 'stick', 'slip')),
            'e_n_used': e, 'magnus_to_spin_scale': scale,
            'conditional_spin_assumptions': dict(ASSUMPTIONS)}
    else:
        raise ValueError('impact_mode must be effective or stick_slip')
    return (state, diagnostic) if return_diagnostics else state


def jacobian_diagnostics(jacobian):
    """Column-scaled singular values; no claim of unique true spin/CL."""
    a = jacobian.toarray() if hasattr(jacobian, 'toarray') else np.asarray(jacobian)
    scale = np.linalg.norm(a, axis=0)
    scaled = a/np.maximum(scale, 1e-15)
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = max(scaled.shape)*np.finfo(float).eps*singular[0]
    rank = int(np.sum(singular > tolerance))
    return {'rows': int(a.shape[0]), 'columns': int(a.shape[1]), 'rank': rank,
        'column_scaled_condition': float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
        'smallest_singular_value': float(singular[-1]),
        'largest_singular_value': float(singular[0]), 'rank_tolerance': float(tolerance),
        'column_norm_min': float(scale.min()),
        'interpretation': 'Local numerical sensitivity of effective parameters; not proof of global identifiability or measured spin'}


def fit_shared_drag_magnus(segments, *, representative_points=48, loss_scale_m=.004,
                           b_limit_per_s=2., starts=(.01, .02, .05), max_nfev=300,
                           fixed_k_drag=None, diagnose=True):
    """Offline training-only fit: one k_D, per-trajectory p0/v0/b_eff.

    Caller owns frozen train membership. Whole flight observations must never
    initialize an online prediction. All starts are scored on this same train
    objective; no validation/test labels participate in coefficient fitting.
    """
    if len(segments) < 2 or b_limit_per_s <= 0:
        raise ValueError('at least two training segments and positive b bound required')
    count = min(representative_points, min(len(t) for t, _ in segments))
    if count < 8:
        raise ValueError('at least eight observations per segment required')
    times, positions, guesses = [], [], []
    for t, pos in segments:
        t, pos = np.asarray(t), np.asarray(pos)
        if np.ptp(t) < .025 or pos.shape != (len(t), 3):
            raise ValueError('flight segment too short or shape invalid')
        idx = np.unique(np.linspace(0, len(t)-1, count).astype(int))
        times.append(t[idx]-t[0]); positions.append(pos[idx])
        guesses.append(np.r_[initial_guess(t, pos), np.zeros(3)])
    times, positions = np.asarray(times), np.asarray(positions)
    n, rows = len(segments), len(segments)*count*3
    free_drag = fixed_k_drag is None
    offset = 1 if free_drag else 0
    lower_states = np.tile(np.r_[np.full(6, -np.inf), np.full(3, -b_limit_per_s)], n)
    upper_states = np.tile(np.r_[np.full(6, np.inf), np.full(3, b_limit_per_s)], n)
    lower = np.r_[0., lower_states] if free_drag else lower_states
    upper = np.r_[.5, upper_states] if free_drag else upper_states
    width = 10 if free_drag else 9
    ri = np.repeat(np.arange(rows), width)
    ci = offset + 9*np.repeat(np.arange(n), count*3)[:, None]+np.arange(9)[None, :]
    if free_drag:
        ci = np.column_stack((np.zeros(rows, dtype=int), ci))
    reports, candidates = [], []
    for start in starts if free_drag else (fixed_k_drag,):
        cache = {}
        x0 = np.r_[start, np.asarray(guesses).ravel()] if free_drag else np.asarray(guesses).ravel()

        def evaluate(x):
            if 'x' in cache and np.array_equal(x, cache['x']):
                return
            k = x[0] if free_drag else fixed_k_drag
            prediction, sensitivity = simulate_batch_with_jacobian(times, x[offset:].reshape(n, 9), k)
            values = np.concatenate((sensitivity[..., 9:10], sensitivity[..., :9]), axis=-1) if free_drag else sensitivity[..., :9]
            cache.update(x=x.copy(), residual=(prediction-positions).ravel(),
                jacobian=csr_matrix((values.ravel(), (ri, ci.ravel())), shape=(rows, len(x))))

        def residual(x):
            evaluate(x)
            return cache['residual'].copy()

        def jacobian(x):
            evaluate(x)
            return cache['jacobian'].copy()

        fit = least_squares(residual, x0, jac=jacobian, bounds=(lower, upper),
            loss='soft_l1', f_scale=loss_scale_m, max_nfev=max_nfev,
            tr_solver='lsmr', tr_options={'atol': 1e-9, 'btol': 1e-9, 'maxiter': 2500},
            x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-7)
        states = fit.x[offset:].reshape(n, 9)
        reports.append({'start_k_drag_per_m': float(start), 'k_drag_per_m': float(fit.x[0] if free_drag else fixed_k_drag),
            'cost': float(fit.cost), 'converged': bool(fit.success), 'nfev': int(fit.nfev),
            'message': str(fit.message), 'b_bound_traces': int(np.any(abs(states[:, 6:]) > b_limit_per_s-1e-5, axis=1).sum())})
        candidates.append(fit)
    qualified = [i for i, f in enumerate(candidates) if f.success]
    chosen = min(qualified or range(len(candidates)), key=lambda i: candidates[i].cost)
    fit = candidates[chosen]
    states = fit.x[offset:].reshape(n, 9)
    k = float(fit.x[0] if free_drag else fixed_k_drag)
    result = {**reports[chosen], 'initial_states': states, 'k_drag_per_m': k,
        'drag_C_D_conditional': drag_coefficient(k), 'multi_start': reports,
        'b_eff_units': '1/s; not RPM', 'b_limit_per_s': float(b_limit_per_s),
        'assumptions': dict(ASSUMPTIONS), 'definition': 'offline reconstruction on training trajectories only',
        'best_start_index': int(chosen)}
    if diagnose:
        _, sensitivity = simulate_batch_with_jacobian(times, states, k)
        values = np.concatenate((sensitivity[..., 9:10], sensitivity[..., :9]), axis=-1) if free_drag else sensitivity[..., :9]
        raw_jac = csr_matrix((values.ravel(), (ri, ci.ravel())), shape=(rows, len(fit.x)))
        result['sensitivity_diagnostics'] = jacobian_diagnostics(raw_jac)
    return result


def fit_conditional_impact(records, parameters=None, *, speed_law=False,
                           mu_starts=(.05, .2, .45, .8)):
    """Fit mu/en to offline v-/v+ labels and fitted b-, under fixed CL slope.

    Each record has v_minus_m_s, v_plus_m_s, b_minus_per_s.  No actual spin is
    observed. Caller filters quality and train membership before invoking.
    Friction starts cross stick/slip regimes; saturated all-stick plateaus can
    have a zero mu derivative. Select by the identical robust training cost,
    never by unweighted RMSE, validation/test error, or a parameter profile.
    """
    if len(records) < 5:
        raise ValueError('at least five training impacts required')
    mu_starts = tuple(float(mu) for mu in mu_starts)
    if not mu_starts or not all(np.isfinite(mu) and 0 < mu < 1 for mu in mu_starts):
        raise ValueError('nonempty finite mu starts strictly inside (0,1) required')
    incoming = np.asarray([np.r_[np.zeros(3), r['v_minus_m_s'], r['b_minus_per_s']] for r in records])
    outgoing = np.asarray([r['v_plus_m_s'] for r in records])
    base = {**ASSUMPTIONS, **(parameters or {}), 'impact_mode': 'stick_slip'}
    reference = float(np.median(-incoming[:, 5]))
    def residual(x):
        p = {**base, 'e_n': x[0], 'mu': x[1], 'normal_speed_reference_m_s': reference,
             'normal_speed_slope': x[2] if speed_law else 0.}
        return (impact_map(incoming, p)[:, 3:6]-outgoing).ravel()
    lo, hi = ([0., 0., -.1], [1., 1., .1]) if speed_law else ([0., 0.], [1., 1.])
    candidates, reports = [], []
    for mu in mu_starts:
        x0 = [.83, mu, 0.] if speed_law else [.83, mu]
        fit = least_squares(residual, x0, bounds=(lo, hi), loss='soft_l1', f_scale=.2,
                            max_nfev=150, x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10)
        p = {**base, 'e_n': float(fit.x[0]), 'mu': float(fit.x[1]),
             'normal_speed_reference_m_s': reference,
             'normal_speed_slope': float(fit.x[2]) if speed_law else 0.}
        _, diag = impact_map(incoming, p, return_diagnostics=True)
        sensitivity = jacobian_diagnostics(fit.jac)
        reports.append({'start_mu': mu, 'e_n': p['e_n'], 'mu': p['mu'],
            'normal_speed_slope': p['normal_speed_slope'],
            'cost': float(fit.cost), 'converged': bool(fit.success),
            'nfev': int(fit.nfev), 'message': str(fit.message),
            'at_parameter_bound': bool(np.any(np.minimum(fit.x-np.asarray(lo), np.asarray(hi)-fit.x) < 1e-5)),
            'training_regimes': {name: int(np.sum(diag['regime'] == name)) for name in ('stick', 'slip')},
            'sensitivity_diagnostics': sensitivity,
            'jacobian_rank': sensitivity['rank']})
        candidates.append(p)
    converged = [i for i, r in enumerate(reports) if r['converged']]
    selected = min(converged or range(len(reports)), key=lambda i: reports[i]['cost'])
    chosen = reports[selected]
    return {**chosen, 'parameters': candidates[selected], 'multi_start': reports,
        'best_start_index': int(selected),
        'selection_objective': 'same training soft_l1 least-squares cost, f_scale=0.2 m/s; converged starts preferred',
        'definition': 'Conditional impact inverse fit given offline incident v/b, fixed linear lift slope and thin shell inertia'}
