import time
import numpy as np
import pandas as pd
import streamlit as st

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve
from scipy.special import gamma

# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="T–τ Opportunistic Inspection Policy", layout="wide")
st.title("T–τ Opportunistic Inspection Policy")
st.caption(
    "Evaluation and optimization of a quasi-periodic opportunistic inspection "
    "policy for a multi-component series system under the delay-time framework."
)

GLOBAL_TOL = 1e-5
RENEWAL_TOL = 1e-10
ALPHA_UPPER = 0.999


# ============================================================
# PROBABILITY FUNCTIONS
# ============================================================

def exponential_pdf(t, lam):
    """PDF of X ~ Exponential(lam), where X is the time to defect arrival."""
    return lam * np.exp(-lam * t)


def weibull_cdf(t, beta, eta):
    """CDF of H ~ Weibull(beta, eta), where H is the delay time to failure."""
    t = np.asarray(t, dtype=float)
    return 1.0 - np.exp(-(np.maximum(t, 0.0) / eta) ** beta)


def opportunity_pdf(w, mu):
    """Density of the waiting time to an external opportunity."""
    return mu * np.exp(-mu * w)


def prob_no_opportunity(tau, mu):
    """Probability of no opportunity in a window of length tau."""
    return float(np.exp(-mu * tau))


def integrate_y(y, x):
    return np.trapezoid(y, x)


# ============================================================
# AUTOMATIC NUMERICAL SETTINGS
# ============================================================

def automatic_settings(quantities, lambda_x, beta_h, eta_h, required_horizon=None):
    quantities = np.asarray(quantities, dtype=float)
    lambda_x = np.asarray(lambda_x, dtype=float)
    beta_h = np.asarray(beta_h, dtype=float)
    eta_h = np.asarray(eta_h, dtype=float)

    mean_x = 1.0 / lambda_x
    mean_h = eta_h * gamma(1.0 + 1.0 / beta_h)
    mean_z = mean_x + mean_h

    total_components = max(float(np.sum(quantities)), 1.0)
    system_scale = float(np.min(mean_z) / max(total_components ** 0.35, 1.0))

    T_lower = max(1e-3, 0.02 * system_scale)
    T_upper = min(max(50.0, 4.0 * system_scale), 10000.0)

    # Tail-based horizon. Using only multiples of the mean can truncate a
    # non-negligible part of Z = X + H, especially for long-tailed Weibull H.
    tail_prob = 1e-8
    x_tail = -np.log(tail_prob) / lambda_x
    h_tail = eta_h * (-np.log(tail_prob)) ** (1.0 / beta_h)

    # If X <= x_tail and H <= h_tail, then Z <= x_tail + h_tail.
    # By the union bound, the omitted component probability is at most
    # approximately 2*tail_prob before discretization error.
    z_tail = x_tail + h_tail

    t_max = max(
        2.2 * T_upper,
        float(np.max(z_tail)),
    )

    if required_horizon is not None:
        t_max = max(t_max, 1.20 * float(required_horizon))
        T_upper = max(T_upper, min(float(required_horizon), 10000.0))

    dt = float(np.clip(system_scale / 2000.0, 0.01, 0.5))
    max_grid_points = 50000
    if t_max / dt > max_grid_points:
        dt = t_max / max_grid_points

    ncomp = int(np.sum(quantities))
    n_quad = 140 if ncomp < 8 else 110 if ncomp < 15 else 80
    n_types = len(quantities)

    return {
        "T_lower": float(T_lower),
        "T_upper": float(T_upper),
        "t_max": float(t_max),
        "dt": float(dt),
        "n_quad": int(n_quad),
        "max_renewal_terms": 500,
        "renewal_tol": RENEWAL_TOL,
        "global_popsize": int(min(14, max(7, 6 + n_types))),
        "global_maxiter": int(min(60, max(25, 22 + 2 * n_types))),
        "global_tol": GLOBAL_TOL,
        "local_maxiter": 1000,
        "alpha_upper": ALPHA_UPPER,
        "system_scale": system_scale,
        "total_components": ncomp,
    }


# ============================================================
# RELIABILITY AND RENEWAL MODEL
# ============================================================

def build_model(quantities, lambda_x, beta_h, eta_h, dt, t_max,
                max_renewal_terms, renewal_tol):
    """
    X_j ~ Exponential(lambda_j): time to defect arrival.
    H_j ~ Weibull(beta_j, eta_j): delay time from defect to failure.
    Z_j = X_j + H_j: total time to component failure.

    F_Z is evaluated as f_X * F_H. This is mathematically equivalent to
    obtaining f_Z = f_X * f_H and then integrating, but is numerically more
    stable when beta < 1 because the Weibull density is singular at zero.

    For a series system:
        phi_j(t) = q_j f_Zj(t) S_Zj(t)^(q_j-1)
                   prod_{k != j} S_Zk(t)^q_k

        f_S(t) = sum_j phi_j(t)

    The cause-specific renewal density is:
        r_j = phi_j + f_S*phi_j + f_S^(2)*phi_j + ...

    and M_j(t) = integral_0^t r_j(u) du.
    """
    start = time.perf_counter()

    quantities = np.asarray(quantities, dtype=float)
    lambda_x = np.asarray(lambda_x, dtype=float)
    beta_h = np.asarray(beta_h, dtype=float)
    eta_h = np.asarray(eta_h, dtype=float)

    n_types = len(quantities)
    t = np.arange(0.0, t_max + dt, dt)
    n_grid = len(t)

    f_z = np.zeros((n_types, n_grid))
    F_z = np.zeros((n_types, n_grid))
    S_z = np.zeros((n_types, n_grid))
    component_masses = []

    for j in range(n_types):
        f_x = exponential_pdf(t, lambda_x[j])
        F_h = weibull_cdf(t, beta_h[j], eta_h[j])

        F_raw = fftconvolve(f_x, F_h)[:n_grid] * dt
        F_raw = np.clip(F_raw, 0.0, 1.0)
        F_z[j] = np.maximum.accumulate(F_raw)

        f_z[j] = np.maximum(np.gradient(F_z[j], dt, edge_order=2), 0.0)
        S_z[j] = np.clip(1.0 - F_z[j], 0.0, 1.0)
        component_masses.append(float(integrate_y(f_z[j], t)))

    phi = np.zeros_like(f_z)
    for j in range(n_types):
        own = quantities[j] * f_z[j] * np.power(
            np.maximum(S_z[j], 1e-15), quantities[j] - 1.0
        )
        others = np.ones_like(t)
        for k in range(n_types):
            if k != j:
                others *= np.power(np.maximum(S_z[k], 1e-15), quantities[k])
        phi[j] = own * others

    f_s = np.sum(phi, axis=0)
    system_first_failure_mass = float(integrate_y(f_s, t))

    # Renewal series: phi_j + f_s*phi_j + f_s^2*phi_j + ...
    cause_density = phi.copy()
    f_power = f_s.copy()

    renewal_stop_term = max_renewal_terms
    renewal_stop_mass = np.nan

    for r in range(1, max_renewal_terms + 1):
        for j in range(n_types):
            cause_density[j] += fftconvolve(f_power, phi[j])[:n_grid] * dt

        f_power_next = fftconvolve(f_power, f_s)[:n_grid] * dt
        added_mass = float(integrate_y(f_power_next, t))
        f_power = f_power_next

        if added_mass < renewal_tol:
            renewal_stop_term = r + 1
            renewal_stop_mass = added_mass
            break

    M_j = np.zeros_like(cause_density)
    for j in range(n_types):
        M_j[j] = cumulative_trapezoid(cause_density[j], t, initial=0.0)

    info = {
        "dt": dt,
        "t_max": t_max,
        "n_grid": n_grid,
        "component_masses": component_masses,
        "system_first_failure_mass": system_first_failure_mass,
        "renewal_stop_term": renewal_stop_term,
        "renewal_stop_mass": renewal_stop_mass,
        "build_time": time.perf_counter() - start,
    }
    return t, M_j, info


# ============================================================
# COST MODEL
# ============================================================

def make_cost_functions(t, M_j_grid, cef, ci, co, cf, mu, n_quad):
    cef = np.asarray(cef, dtype=float)
    n_types = len(cef)

    def M_at(a):
        a = float(np.clip(a, 0.0, t[-1]))
        return np.array([np.interp(a, t, M_j_grid[j]) for j in range(n_types)])

    def failure_cost_until(a):
        # Each failure caused by type j generates base cost Cf plus type-specific CEF_j.
        return float(np.sum(M_at(a) * (cf + cef)))

    def ec_i(T, tau):
        q = prob_no_opportunity(tau, mu)
        no_opp = ci + failure_cost_until(T)
        if tau <= 0.0:
            return float(no_opp)

        w = np.linspace(0.0, tau, n_quad)
        ages = T - tau + w
        integrand = opportunity_pdf(w, mu) * np.array(
            [co + failure_cost_until(a) for a in ages]
        )
        return float(q * no_opp + integrate_y(integrand, w))

    def ev_i(T, tau):
        q = prob_no_opportunity(tau, mu)
        if tau <= 0.0:
            return float(T)
        w = np.linspace(0.0, tau, n_quad)
        ages = T - tau + w
        return float(q * T + integrate_y(opportunity_pdf(w, mu) * ages, w))

    def ec_o(T, tau):
        if tau <= 0.0:
            return float(ci + failure_cost_until(T))

        q = prob_no_opportunity(tau, mu)
        pi_o = 1.0 - q
        w = np.linspace(0.0, tau, n_quad)
        y = np.linspace(0.0, tau, n_quad)
        fw = opportunity_pdf(w, mu)

        # o -> i
        integrand_oi = fw * np.array(
            [ci + failure_cost_until(T + tau - wi) for wi in w]
        )
        cost_oi = q * integrate_y(integrand_oi, w)

        # o -> o
        fy = opportunity_pdf(y, mu)
        matrix = np.zeros((len(w), len(y)))
        for i, wi in enumerate(w):
            ages = T + y - wi
            matrix[i, :] = fw[i] * fy * np.array(
                [co + failure_cost_until(a) for a in ages]
            )
        cost_oo = integrate_y(np.trapezoid(matrix, y, axis=1), w)

        return float((cost_oi + cost_oo)/pi_o)

    def ev_o(T, tau):
        if tau <= 0.0:
            return float(T)

        q = prob_no_opportunity(tau, mu)
        pi_o = 1.0 - q
        w = np.linspace(0.0, tau, n_quad)
        y = np.linspace(0.0, tau, n_quad)
        fw = opportunity_pdf(w, mu)

        dur_oi = q * integrate_y(fw * (T + tau - w), w)

        fy = opportunity_pdf(y, mu)
        matrix = np.zeros((len(w), len(y)))
        for i, wi in enumerate(w):
            matrix[i, :] = fw[i] * fy * (T + y - wi)
        dur_oo = integrate_y(np.trapezoid(matrix, y, axis=1), w)

        return float((dur_oi + dur_oo)/pi_o)

    def performance(T, tau):
        if T <= 0:
            raise ValueError("T must be greater than zero.")
        if tau < 0:
            raise ValueError("tau cannot be negative.")
        if tau >= T:
            raise ValueError("tau must be smaller than T.")
        if T + tau > t[-1]:
            raise ValueError("Numerical horizon is too short for this T and tau.")

        pi_i = prob_no_opportunity(tau, mu)
        pi_o = 1.0 - pi_i

        # ------------------------------------------------------------
        # Transition-probability consistency check
        # ------------------------------------------------------------
        # I -> I: no opportunity occurs in a window of length tau.
        P_II = pi_i

        # I -> O: at least one opportunity occurs in the window.
        # This is intentionally evaluated by numerical integration so that
        # the check also detects quadrature inaccuracies.
        if tau <= 0.0:
            P_IO = 0.0
            P_OI = np.nan
            P_OO = np.nan
        else:
            w_check = np.linspace(0.0, tau, n_quad)
            opportunity_mass = integrate_y(opportunity_pdf(w_check, mu), w_check)
            P_IO = float(opportunity_mass)

            # The O-state terms in the model are written with the raw
            # opportunity density. Their total raw mass is pi_o. Dividing
            # the O->I and O->O masses by that O-state mass gives the
            # corresponding transition probabilities, without changing the
            # cost expressions used by the model.
            raw_P_OI = pi_i * opportunity_mass
            raw_P_OO = opportunity_mass * opportunity_mass
            if opportunity_mass > 1e-14:
                P_OI = float(raw_P_OI / opportunity_mass)
                P_OO = float(raw_P_OO / opportunity_mass)
            else:
                P_OI = np.nan
                P_OO = np.nan

        sum_I = float(P_II + P_IO)
        sum_O = float(P_OI + P_OO) if np.isfinite(P_OI) and np.isfinite(P_OO) else np.nan

        EC_i = ec_i(T, tau)
        EV_i = ev_i(T, tau)

        if pi_o < 1e-14:
            EC_o = np.nan
            EV_o = np.nan
            EC = EC_i
            EV = EV_i
        else:
            EC_o = ec_o(T, tau)
            EV_o = ev_o(T, tau)
            EC = pi_i * EC_i + pi_o * EC_o
            EV = pi_i * EV_i + pi_o * EV_o

        return {
            "T": float(T),
            "tau": float(tau),
            "tau/T": float(tau / T),
            "pi_i": float(pi_i),
            "pi_o": float(pi_o),
            "P_II": float(P_II),
            "P_IO": float(P_IO),
            "P_OI": float(P_OI) if np.isfinite(P_OI) else np.nan,
            "P_OO": float(P_OO) if np.isfinite(P_OO) else np.nan,
            "P_II + P_IO": sum_I,
            "P_OI + P_OO": sum_O,
            "EC_i": float(EC_i),
            "EV_i": float(EV_i),
            "EC_o": float(EC_o) if np.isfinite(EC_o) else np.nan,
            "EV_o": float(EV_o) if np.isfinite(EV_o) else np.nan,
            "Expected cycle cost": float(EC),
            "Expected cycle duration": float(EV),
            "Long-run cost rate": float(EC / EV),
        }

    def cost_rate(T, tau):
        if T <= 0 or tau < 0 or tau >= T or T + tau > t[-1]:
            return np.inf
        try:
            return performance(T, tau)["Long-run cost rate"]
        except Exception:
            return np.inf

    return performance, cost_rate


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_policy(cost_rate, settings, progress_container=None):
    rows = []
    counter = 0
    best = np.inf

    def objective(x, stage):
        nonlocal counter, best
        T = float(x[0])
        alpha = float(x[1])
        tau = alpha * T
        value = cost_rate(T, tau)

        counter += 1
        best = min(best, value)
        rows.append({
            "Stage": stage,
            "Evaluation": counter,
            "T": T,
            "tau": tau,
            "tau/T": alpha,
            "Cost rate": value,
            "Best cost rate": best,
        })

        if progress_container is not None and counter % 5 == 0:
            progress_container.dataframe(
                pd.DataFrame(rows).tail(25),
                use_container_width=True,
                hide_index=True,
            )
        return value

    bounds = [
        (settings["T_lower"], settings["T_upper"]),
        (0.0, settings["alpha_upper"]),
    ]

    global_result = differential_evolution(
        lambda x: objective(x, "Global"),
        bounds=bounds,
        seed=123,
        popsize=settings["global_popsize"],
        maxiter=settings["global_maxiter"],
        tol=settings["global_tol"],
        polish=False,
        updating="immediate",
        workers=1,
    )

    local_result = minimize(
        lambda x: objective(x, "Local"),
        x0=global_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": settings["local_maxiter"], "ftol": 1e-12},
    )

    candidates = [(global_result.fun, global_result.x), (local_result.fun, local_result.x)]
    best_fun, best_x = min(candidates, key=lambda z: z[0])

    T_star = float(best_x[0])
    alpha_star = float(best_x[1])
    tau_star = alpha_star * T_star

    return {
        "T_star": T_star,
        "tau_star": tau_star,
        "alpha_star": alpha_star,
        "C_star": float(best_fun),
        "log": pd.DataFrame(rows),
    }


# ============================================================
# INTERFACE
# ============================================================

st.header("1. Analysis mode")
mode = st.radio(
    "Choose what the software should do",
    ["Evaluate a specified T and τ", "Optimize T and τ"],
    captions=[
        "Enter T and τ and obtain the long-run performance directly.",
        "Let the software search for the values of T and τ that minimize the long-run cost rate.",
    ],
)

st.divider()
st.header("2. General parameters")
st.markdown(
    """
Use the same time unit for all reliability and policy parameters and the same
monetary unit for all costs.

**Ci** is the fixed cost of a scheduled inspection/intervention.  
**Co** is the fixed cost of an opportunistic inspection/intervention.  
**Cf** is the base corrective cost associated with every system failure.  
**μ** is the arrival rate of external opportunities. Opportunities are assumed
to follow a homogeneous Poisson process, so the waiting time between successive
opportunities is exponentially distributed with mean **1/μ**.
"""
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    ci = st.number_input("Scheduled intervention cost (Ci)", min_value=0.0, value=None, placeholder="Enter Ci")
with c2:
    co = st.number_input("Opportunistic intervention cost (Co)", min_value=0.0, value=None, placeholder="Enter Co")
with c3:
    cf = st.number_input("Base failure cost (Cf)", min_value=0.0, value=None, placeholder="Enter Cf")
with c4:
    mu = st.number_input(
        "Opportunity arrival rate (μ)", min_value=0.0, value=None,
        placeholder="Enter μ", format="%.8f",
        help="Rate of the homogeneous Poisson process that generates external opportunities."
    )

st.divider()
st.header("3. Component structure and delay-time parameters")
st.markdown(
    """
For component type **j**:

- **Quantity qj**: number of identical components of that type in the series system.
- **E[Xj]**: mean time from renewal/replacement until a detectable defect arrives. The time to defect **Xj** is exponentially distributed. The model internally uses `λj = 1/E[Xj]`, so `Xj ~ Exponential(λj)`.
- **βj**: Weibull shape parameter for **Hj**, the delay time between defect arrival and functional failure.
- **ηj**: Weibull scale parameter for **Hj**. Thus `Hj ~ Weibull(βj, ηj)`.
- The total failure time is **Zj = Xj + Hj**.
- **CEFj**: additional failure consequence cost when the system failure is caused by component type j. The total cost of such a failure is `Cf + CEFj`.
"""
)

n_types = st.number_input(
    "Number of distinct component types",
    min_value=1,
    max_value=20,
    value=None,
    step=1,
    placeholder="Enter number of component types",
)

quantities, mean_x_input, beta_h, eta_h, cef = [], [], [], [], []

if n_types is not None:
    for i in range(int(n_types)):
        with st.expander(f"Component type {i+1}", expanded=True):
            a, b, c, d, e = st.columns(5)
            with a:
                q = st.number_input(
                    f"Quantity q{i+1}", min_value=1, value=None, step=1,
                    placeholder="q", key=f"q_{i}"
                )
            with b:
                mean_x = st.number_input(
                    f"Mean time to defect E[X{i+1}]", min_value=0.0,
                    value=None, placeholder="Mean X", format="%.6f", key=f"mean_x_{i}",
                    help=(f"Mean time until defect arrival for component type {i+1}. "
                          f"X{i+1} is exponentially distributed and the model uses "
                          f"λ{i+1} = 1 / E[X{i+1}].")
                )
            with c:
                beta = st.number_input(
                    f"Weibull shape β{i+1} for H{i+1}", min_value=0.0,
                    value=None, placeholder="β", format="%.6f", key=f"beta_{i}",
                    help=f"H{i+1} is the delay time from defect arrival to failure."
                )
            with d:
                eta = st.number_input(
                    f"Weibull scale η{i+1} for H{i+1}", min_value=0.0,
                    value=None, placeholder="η", format="%.6f", key=f"eta_{i}"
                )
            with e:
                ce = st.number_input(
                    f"Extra failure cost CEF{i+1}", min_value=0.0,
                    value=None, placeholder="CEF", key=f"cef_{i}"
                )

            quantities.append(q)
            mean_x_input.append(mean_x)
            beta_h.append(beta)
            eta_h.append(eta)
            cef.append(ce)

st.divider()
st.header("4. Policy variables")

T_user = tau_user = None
if mode == "Evaluate a specified T and τ":
    p1, p2 = st.columns(2)
    with p1:
        T_user = st.number_input(
            "Scheduled inspection interval (T)", min_value=0.0, value=None,
            placeholder="Enter T"
        )
    with p2:
        tau_user = st.number_input(
            "Opportunistic inspection window (τ)", min_value=0.0, value=None,
            placeholder="Enter τ", help="The policy requires 0 ≤ τ < T."
        )
else:
    st.info(
        "T and τ are decision variables. The software searches globally and then "
        "performs a bounded local refinement. The constraint 0 ≤ τ < T is handled "
        "through α = τ/T."
    )

run_label = "Evaluate policy" if mode == "Evaluate a specified T and τ" else "Optimize policy"
run = st.button(run_label, type="primary")


# ============================================================
# EXECUTION
# ============================================================

if run:
    errors = []

    if any(v is None for v in [ci, co, cf, mu]):
        errors.append("Complete all general parameters.")
    elif mu <= 0:
        errors.append("μ must be greater than zero.")

    if n_types is None:
        errors.append("Enter the number of component types.")
    else:
        all_component_values = quantities + mean_x_input + beta_h + eta_h + cef
        if any(v is None for v in all_component_values):
            errors.append("Complete all component parameters.")
        else:
            if any(v <= 0 for v in mean_x_input):
                errors.append("Every mean time to defect E[Xj] must be greater than zero.")
            if any(v <= 0 for v in beta_h):
                errors.append("Every Weibull shape βj must be greater than zero.")
            if any(v <= 0 for v in eta_h):
                errors.append("Every Weibull scale ηj must be greater than zero.")

    if mode == "Evaluate a specified T and τ":
        if T_user is None or tau_user is None:
            errors.append("Enter both T and τ.")
        elif T_user <= 0:
            errors.append("T must be greater than zero.")
        elif tau_user >= T_user:
            errors.append("τ must be smaller than T.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    quantities_np = np.asarray(quantities, dtype=float)
    mean_x_np = np.asarray(mean_x_input, dtype=float)
    lambda_np = 1.0 / mean_x_np
    beta_np = np.asarray(beta_h, dtype=float)
    eta_np = np.asarray(eta_h, dtype=float)
    cef_np = np.asarray(cef, dtype=float)

    required_horizon = None
    if mode == "Evaluate a specified T and τ":
        required_horizon = float(T_user + tau_user)

    settings = automatic_settings(
        quantities_np, lambda_np, beta_np, eta_np,
        required_horizon=required_horizon,
    )

    with st.status("Building reliability and renewal model...", expanded=False) as status:
        t, M_j, info = build_model(
            quantities_np, lambda_np, beta_np, eta_np,
            settings["dt"], settings["t_max"],
            settings["max_renewal_terms"], settings["renewal_tol"],
        )
        status.update(label="Reliability and renewal model completed.", state="complete")

    performance, cost_rate = make_cost_functions(
        t, M_j, cef_np, float(ci), float(co), float(cf), float(mu), settings["n_quad"]
    )

    def show_numerical_checks(info, result, tolerance=1e-4):
        with st.expander("Numerical checks", expanded=False):
            st.markdown("#### Distribution-mass checks")

            masses = np.asarray(info["component_masses"], dtype=float)
            mass_errors = np.abs(masses - 1.0)
            check_df = pd.DataFrame({
                "Component type": [f"Type {i+1}" for i in range(int(n_types))],
                "Integral of f_Z over numerical grid": masses,
                "Absolute error from 1": mass_errors,
                "Status": np.where(mass_errors <= tolerance, "OK", "Check"),
            })
            st.dataframe(check_df, use_container_width=True, hide_index=True)

            system_mass = float(info["system_first_failure_mass"])
            system_error = abs(system_mass - 1.0)
            st.write(
                f"System first-failure density mass over grid: {system_mass:.10f} "
                f"(error from 1 = {system_error:.3e})"
            )

            st.markdown("#### Transition-probability consistency checks")

            sum_I = result["P_II + P_IO"]
            sum_O = result["P_OI + P_OO"]

            probability_df = pd.DataFrame({
                "Transition": ["P_II", "P_IO", "P_OI", "P_OO"],
                "Probability": [
                    result["P_II"], result["P_IO"],
                    result["P_OI"], result["P_OO"],
                ],
            })
            st.dataframe(probability_df, use_container_width=True, hide_index=True)

            row_data = [
                {
                    "Probability sum": "P_II + P_IO",
                    "Value": sum_I,
                    "Absolute error from 1": abs(sum_I - 1.0),
                    "Status": "OK" if abs(sum_I - 1.0) <= tolerance else "Check",
                }
            ]

            if np.isfinite(sum_O):
                row_data.append({
                    "Probability sum": "P_OI + P_OO",
                    "Value": sum_O,
                    "Absolute error from 1": abs(sum_O - 1.0),
                    "Status": "OK" if abs(sum_O - 1.0) <= tolerance else "Check",
                })
            else:
                row_data.append({
                    "Probability sum": "P_OI + P_OO",
                    "Value": np.nan,
                    "Absolute error from 1": np.nan,
                    "Status": "N/A (τ = 0)",
                })

            st.dataframe(pd.DataFrame(row_data), use_container_width=True, hide_index=True)

            all_mass_ok = bool(np.all(mass_errors <= tolerance))
            system_ok = system_error <= tolerance
            prob_I_ok = abs(sum_I - 1.0) <= tolerance
            prob_O_ok = (result["tau"] <= 0.0) or (
                np.isfinite(sum_O) and abs(sum_O - 1.0) <= tolerance
            )

            if all_mass_ok and system_ok and prob_I_ok and prob_O_ok:
                st.success(
                    f"All numerical checks are within tolerance ({tolerance:g})."
                )
            else:
                warnings = []
                if not all_mass_ok:
                    warnings.append("one or more component failure-time densities do not integrate sufficiently close to 1")
                if not system_ok:
                    warnings.append("the system first-failure density does not integrate sufficiently close to 1")
                if not prob_I_ok or not prob_O_ok:
                    warnings.append("one or more transition-probability rows do not sum sufficiently close to 1")
                st.warning(
                    "Numerical check warning: " + "; ".join(warnings) + "."
                )

            st.caption(
                f"Grid: {info['n_grid']} points; dt = {info['dt']:.6g}; "
                f"horizon = {info['t_max']:.6g}; renewal summation stopped at "
                f"convolution order {info['renewal_stop_term']}."
            )

    if mode == "Evaluate a specified T and τ":
        result = performance(float(T_user), float(tau_user))

        st.subheader("Policy performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("T", f"{result['T']:.6f}")
        m2.metric("τ", f"{result['tau']:.6f}")
        m3.metric("τ/T", f"{result['tau/T']:.6f}")
        m4.metric("Long-run cost rate", f"{result['Long-run cost rate']:.8f}")

        details = pd.DataFrame({
            "Measure": [
                "Probability of scheduled-state cycle (πi)",
                "Probability of opportunistic-state cycle (πo)",
                "Expected cost from scheduled state (ECi)",
                "Expected duration from scheduled state (EVi)",
                "Expected cost from opportunistic state (ECo)",
                "Expected duration from opportunistic state (EVo)",
                "Overall expected cycle cost",
                "Overall expected cycle duration",
                "Long-run expected cost rate",
            ],
            "Value": [
                result["pi_i"], result["pi_o"], result["EC_i"], result["EV_i"],
                result["EC_o"], result["EV_o"], result["Expected cycle cost"],
                result["Expected cycle duration"], result["Long-run cost rate"],
            ]
        })
        st.dataframe(details, use_container_width=True, hide_index=True)
        show_numerical_checks(info, result)

    else:
        st.subheader("Optimization progress")
        progress = st.empty()

        with st.status("Optimizing T and τ...", expanded=False) as status:
            opt = optimize_policy(cost_rate, settings, progress)
            status.update(label="Optimization completed.", state="complete")

        result = performance(opt["T_star"], opt["tau_star"])

        st.subheader("Optimal policy")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal T*", f"{opt['T_star']:.6f}")
        m2.metric("Optimal τ*", f"{opt['tau_star']:.6f}")
        m3.metric("τ*/T*", f"{opt['alpha_star']:.6f}")
        m4.metric("Minimum cost rate", f"{opt['C_star']:.8f}")

        st.markdown("#### Performance at the optimum")
        details = pd.DataFrame({
            "Measure": [
                "πi", "πo", "ECi", "EVi", "ECo", "EVo",
                "Overall expected cycle cost", "Overall expected cycle duration",
                "Long-run expected cost rate",
            ],
            "Value": [
                result["pi_i"], result["pi_o"], result["EC_i"], result["EV_i"],
                result["EC_o"], result["EV_o"], result["Expected cycle cost"],
                result["Expected cycle duration"], result["Long-run cost rate"],
            ]
        })
        st.dataframe(details, use_container_width=True, hide_index=True)
        show_numerical_checks(info, result)

        with st.expander("Tested solutions", expanded=False):
            st.dataframe(opt["log"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download tested solutions as CSV",
                data=opt["log"].to_csv(index=False).encode("utf-8"),
                file_name="T_tau_optimization_log.csv",
                mime="text/csv",
            )
