"""
T-tau Opportunistic Inspection Model
====================================

Streamlit app for optimizing a quasi-periodic opportunistic inspection policy
for multi-component systems under the delay-time framework.

How to run locally
------------------
1. Install dependencies:
       pip install -r requirements.txt

2. Run:
       streamlit run app.py

Main assumptions
----------------
- Series system configuration.
- The system may contain different component types.
- Each component type may represent several identical components.
- Time-to-defect:
       X_j ~ Exponential(lambda_j)
- Delay-time:
       H_j ~ Weibull(beta_j, eta_j)
- Time-to-failure:
       Z_j = X_j + H_j
- Exogenous opportunities arrive according to a Homogeneous Poisson Process.
- The policy is defined by:
       T   = scheduled inspection interval
       tau = opportunistic inspection window
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="T-tau Opportunistic Inspection Model",
    layout="wide",
)

st.title("T–τ Opportunistic Inspection Model")
st.caption(
    "Quasi-periodic inspection and preventive maintenance optimization "
    "for multi-component systems under the delay-time framework."
)


# ============================================================
# ADAPTIVE NUMERICAL AND OPTIMIZATION SETTINGS
# ============================================================
# These parameters are automatically adjusted from the user's input data.
# The user does not need to configure numerical/optimization details.

ALPHA_UPPER = 0.95
GLOBAL_TOL = 1e-5
RENEWAL_TOL = 1e-11


def automatic_settings(quantities, lambda_x, beta_h, eta_h):
    """
    Choose numerical and optimization settings from the input parameters.

    The goal is to keep the interface simple while still adapting the
    computation to the scale of the reliability data.

    Main idea:
    - Use the mean of X_j plus the mean of H_j as a characteristic life scale.
    - Set the T search interval around that scale.
    - Choose dt as a fraction of the scale, bounded to avoid excessive runtime.
    - Use more global iterations for larger models.
    """

    quantities = np.asarray(quantities, dtype=float)
    lambda_x = np.asarray(lambda_x, dtype=float)
    beta_h = np.asarray(beta_h, dtype=float)
    eta_h = np.asarray(eta_h, dtype=float)

    # Mean of the exponential time-to-defect.
    mean_x = 1.0 / lambda_x

    # Mean of Weibull delay-time:
    # E[H] = eta * Gamma(1 + 1 / beta).
    from scipy.special import gamma
    mean_h = eta_h * gamma(1.0 + 1.0 / beta_h)

    mean_z = mean_x + mean_h

    # For a series system with many components, the system characteristic
    # time is shorter than the average component life. This approximation is
    # only used to define a safe optimization range.
    total_components = max(float(np.sum(quantities)), 1.0)
    system_scale = float(np.min(mean_z) / max(total_components ** 0.35, 1.0))

    # Keep bounds broad enough to allow the optimizer to explore.
    T_lower = max(1.0, 0.02 * system_scale)
    T_upper = max(50.0, 4.0 * system_scale)

    # Avoid extreme upper bounds that may make the app too slow on Streamlit Cloud.
    T_upper = min(T_upper, 10000.0)

    # Time grid horizon. Needs to cover T + tau and enough renewal mass.
    t_max = max(3.0 * T_upper, 2.5 * float(np.max(mean_z)), 10.0 * float(np.max(eta_h)))

    # Adaptive dt: approximately 1200-8000 grid points depending on scale.
    # Smaller dt improves accuracy but increases runtime.
    dt = system_scale / 1000.0
    dt = float(np.clip(dt, 0.05, 2.0))

    # If the grid would still be too large, relax dt.
    max_grid_points = 12000
    estimated_points = t_max / dt
    if estimated_points > max_grid_points:
        dt = t_max / max_grid_points

    # Quadrature points for opportunity integrations.
    n_quad = 100
    if total_components >= 8:
        n_quad = 80
    if total_components >= 15:
        n_quad = 60

    # Renewal settings.
    max_renewal_terms = 500

    # Optimization effort grows mildly with the number of component types.
    n_types = len(quantities)
    global_popsize = int(min(12, max(6, 5 + n_types)))
    global_maxiter = int(min(50, max(20, 18 + 2 * n_types)))
    local_maxiter = 800

    return {
        "T_lower": float(T_lower),
        "T_upper": float(T_upper),
        "t_max": float(t_max),
        "dt": float(dt),
        "n_quad": int(n_quad),
        "max_renewal_terms": int(max_renewal_terms),
        "renewal_tol": float(RENEWAL_TOL),
        "global_popsize": int(global_popsize),
        "global_maxiter": int(global_maxiter),
        "global_tol": float(GLOBAL_TOL),
        "local_maxiter": int(local_maxiter),
        "alpha_upper": float(ALPHA_UPPER),
        "system_scale": float(system_scale),
        "total_components": int(np.sum(quantities)),
    }


# ============================================================
# BASIC PROBABILITY FUNCTIONS
# ============================================================

def integrate_y(y, x):
    """
    Numerical integration wrapper compatible with current NumPy versions.

    np.trapz was removed in recent NumPy versions. np.trapezoid is the
    supported replacement.
    """
    return np.trapezoid(y, x)


def weibull_pdf(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Return the Weibull probability density function."""
    f = np.zeros_like(t, dtype=float)
    positive = t > 0

    f[positive] = (
        (beta / eta)
        * (t[positive] / eta) ** (beta - 1.0)
        * np.exp(-(t[positive] / eta) ** beta)
    )

    return f


def exponential_pdf(t: np.ndarray, lam: float) -> np.ndarray:
    """Return the exponential probability density function."""
    return lam * np.exp(-lam * t)


def opportunity_pdf(w: np.ndarray | float, mu: float) -> np.ndarray | float:
    """Return the raw density f_w(w) = mu exp(-mu w)."""
    return mu * np.exp(-mu * w)


def prob_no_opportunity(tau: float, mu: float) -> float:
    """Return the probability of no opportunity in a window of length tau."""
    return float(np.exp(-mu * tau))


# ============================================================
# NUMERICAL MODEL CONSTRUCTION
# ============================================================

def build_model(
    quantities: np.ndarray,
    lambda_x: np.ndarray,
    beta_h: np.ndarray,
    eta_h: np.ndarray,
    dt: float,
    t_max: float,
    max_renewal_terms: int,
    renewal_tol: float,
):
    """
    Build all numerical curves required by the model.

    The system is assumed to be in series. Several identical components can
    be represented by a single component type with quantity q_j.

    Cause-specific first system failure density:
        phi_j(t) = q_j f_j(t) S_j(t)^(q_j-1)
                   product_{k != j} S_k(t)^q_k

    Renewal by cause:
        r_j(t) = phi_j(t) + f_s * phi_j(t) + f_s^(2) * phi_j(t) + ...
        M_j(t) = integral_0^t r_j(u) du
    """

    start = time.perf_counter()

    quantities = np.asarray(quantities, dtype=float)
    lambda_x = np.asarray(lambda_x, dtype=float)
    beta_h = np.asarray(beta_h, dtype=float)
    eta_h = np.asarray(eta_h, dtype=float)

    n_types = len(quantities)

    t = np.arange(0.0, t_max + dt, dt)
    n_grid = len(t)

    f_z = np.zeros((n_types, n_grid), dtype=float)
    f_z_cdf = np.zeros((n_types, n_grid), dtype=float)
    s_z = np.zeros((n_types, n_grid), dtype=float)

    component_masses = []

    for j in range(n_types):
        f_x = exponential_pdf(t, lambda_x[j])
        f_h = weibull_pdf(t, beta_h[j], eta_h[j])

        f_z[j] = fftconvolve(f_x, f_h)[:n_grid] * dt
        f_z_cdf[j] = cumulative_trapezoid(f_z[j], t, initial=0.0)
        s_z[j] = np.clip(1.0 - f_z_cdf[j], 0.0, 1.0)

        component_masses.append(float(integrate_y(f_z[j], t)))

    # Cause-specific first system failure densities.
    phi = np.zeros_like(f_z)

    for j in range(n_types):
        own_type_term = (
            quantities[j]
            * f_z[j]
            * np.power(np.maximum(s_z[j], 1e-15), quantities[j] - 1.0)
        )

        survival_other_types = np.ones_like(t, dtype=float)

        for k in range(n_types):
            if k != j:
                survival_other_types *= np.power(
                    np.maximum(s_z[k], 1e-15),
                    quantities[k],
                )

        phi[j] = own_type_term * survival_other_types

    f_s = np.sum(phi, axis=0)
    system_first_failure_mass = float(integrate_y(f_s, t))

    # Renewal calculation by failure cause.
    cause_density = phi.copy()
    f_power = f_s.copy()

    renewal_stop_term = max_renewal_terms
    renewal_stop_mass = np.nan
    progress_messages = []

    for r in range(1, max_renewal_terms + 1):
        for j in range(n_types):
            cause_density[j] += fftconvolve(f_power, phi[j])[:n_grid] * dt

        f_power_next = fftconvolve(f_power, f_s)[:n_grid] * dt
        added_mass = float(integrate_y(f_power_next, t))
        f_power = f_power_next

        if r <= 5 or r % 25 == 0:
            progress_messages.append(
                f"Renewal term {r}: added mass = {added_mass:.3e}"
            )

        if added_mass < renewal_tol:
            renewal_stop_term = r
            renewal_stop_mass = added_mass
            progress_messages.append(
                f"Renewal stopped at term {r}. Added mass = {added_mass:.3e}"
            )
            break

    mj_grid = np.zeros_like(cause_density)

    for j in range(n_types):
        mj_grid[j] = cumulative_trapezoid(cause_density[j], t, initial=0.0)

    elapsed = time.perf_counter() - start

    info = {
        "dt": dt,
        "t_max": t_max,
        "n_grid": n_grid,
        "component_masses": component_masses,
        "system_first_failure_mass": system_first_failure_mass,
        "renewal_stop_term": renewal_stop_term,
        "renewal_stop_mass": renewal_stop_mass,
        "build_time": elapsed,
        "progress_messages": progress_messages,
    }

    return t, mj_grid, info


# ============================================================
# COST FUNCTIONS
# ============================================================

def make_cost_functions(
    t: np.ndarray,
    mj_grid: np.ndarray,
    cef: np.ndarray,
    ci: float,
    co: float,
    cf: float,
    mu: float,
    n_quad: int,
):
    """Create model cost functions for a fixed numerical grid."""

    cef = np.asarray(cef, dtype=float)
    n_types = len(cef)

    def mj_at(a: float) -> np.ndarray:
        """Interpolate M_j(a) for all component types."""
        a = np.clip(a, 0.0, t[-1])
        return np.array([np.interp(a, t, mj_grid[j]) for j in range(n_types)])

    def failure_cost_until(a: float) -> float:
        """Return expected failure-related cost accumulated until age a."""
        m = mj_at(a)
        return float(np.sum(m * (cf + cef)))

    def ec_i(T: float, tau: float) -> float:
        """Expected cost of a cycle beginning at a scheduled intervention."""
        q = prob_no_opportunity(tau, mu)
        no_opp = ci + failure_cost_until(T)

        if tau <= 0.0:
            return float(no_opp)

        w = np.linspace(0.0, tau, n_quad)
        ages = T - tau + w

        integrand = np.array(
            [
                opportunity_pdf(wi, mu) * (co + failure_cost_until(ai))
                for wi, ai in zip(w, ages)
            ]
        )

        return float(q * no_opp + integrate_y(integrand, w))

    def ev_i(T: float, tau: float) -> float:
        """Expected duration of a cycle beginning at a scheduled intervention."""
        q = prob_no_opportunity(tau, mu)

        if tau <= 0.0:
            return float(T)

        w = np.linspace(0.0, tau, n_quad)
        ages = T - tau + w

        return float(q * T + integrate_y(opportunity_pdf(w, mu) * ages, w))

    def ec_o(T: float, tau: float) -> float:
        """Expected cost of a cycle beginning at an opportunistic intervention."""
        q = prob_no_opportunity(tau, mu)

        if tau <= 0.0:
            return float(ci + failure_cost_until(T))

        w = np.linspace(0.0, tau, n_quad)
        y = np.linspace(0.0, tau, n_quad)

        # Transition o -> i
        integrand_oi = np.array(
            [
                opportunity_pdf(wi, mu)
                * (ci + failure_cost_until(T + tau - wi))
                for wi in w
            ]
        )
        cost_oi = q * integrate_y(integrand_oi, w)

        # Transition o -> o
        matrix = np.zeros((len(w), len(y)), dtype=float)

        for i, wi in enumerate(w):
            fwi = opportunity_pdf(wi, mu)

            for k, yk in enumerate(y):
                age = T + yk - wi
                matrix[i, k] = (
                    fwi
                    * opportunity_pdf(yk, mu)
                    * (co + failure_cost_until(age))
                )

        inner = np.trapezoid(matrix, y, axis=1)
        cost_oo = integrate_y(inner, w)

        return float(cost_oi + cost_oo)

    def ev_o(T: float, tau: float) -> float:
        """Expected duration of a cycle beginning at an opportunistic intervention."""
        q = prob_no_opportunity(tau, mu)

        if tau <= 0.0:
            return float(T)

        w = np.linspace(0.0, tau, n_quad)
        y = np.linspace(0.0, tau, n_quad)

        # Transition o -> i
        integrand_oi = opportunity_pdf(w, mu) * (T + tau - w)
        dur_oi = q * integrate_y(integrand_oi, w)

        # Transition o -> o
        matrix = np.zeros((len(w), len(y)), dtype=float)

        for i, wi in enumerate(w):
            fwi = opportunity_pdf(wi, mu)

            for k, yk in enumerate(y):
                matrix[i, k] = (
                    fwi
                    * opportunity_pdf(yk, mu)
                    * (T + yk - wi)
                )

        inner = np.trapezoid(matrix, y, axis=1)
        dur_oo = integrate_y(inner, w)

        return float(dur_oi + dur_oo)

    def cost_rate(T: float, tau: float) -> float:
        """Return the long-run expected cost-rate."""
        if T <= 0.0 or tau < 0.0 or tau >= T:
            return np.inf

        if T + tau > t[-1]:
            return np.inf

        pi_i = prob_no_opportunity(tau, mu)
        pi_o = 1.0 - pi_i

        ec_cycle = pi_i * ec_i(T, tau) + pi_o * ec_o(T, tau)
        ev_cycle = pi_i * ev_i(T, tau) + pi_o * ev_o(T, tau)

        if ev_cycle <= 0.0:
            return np.inf

        return float(ec_cycle / ev_cycle)

    return cost_rate, ec_i, ev_i, ec_o, ev_o


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_policy(
    cost_rate,
    settings,
    progress_container=None,
):
    """Run global search followed by local refinement."""

    log_rows = []
    eval_counter = {"Global": 0, "Local": 0}

    best = {
        "value": np.inf,
        "T": None,
        "tau": None,
        "alpha": None,
    }

    def add_log(stage: str, T: float, alpha: float, value: float, elapsed: float):
        tau = alpha * T

        if value < best["value"]:
            best["value"] = value
            best["T"] = T
            best["tau"] = tau
            best["alpha"] = alpha

        row = {
            "Stage": stage,
            "Evaluation": eval_counter[stage],
            "T": T,
            "tau": tau,
            "tau/T": alpha,
            "Cost-rate": value,
            "Best cost-rate so far": best["value"],
            "Evaluation time (s)": elapsed,
        }

        log_rows.append(row)

        if progress_container is not None:
            progress_container.dataframe(
                pd.DataFrame(log_rows).tail(25),
                use_container_width=True,
                hide_index=True,
            )

    def global_objective(x):
        eval_counter["Global"] += 1

        T = float(x[0])
        alpha = float(x[1])
        tau = alpha * T

        start = time.perf_counter()
        value = cost_rate(T, tau)
        elapsed = time.perf_counter() - start

        add_log("Global", T, alpha, value, elapsed)
        return value

    bounds = [
        (settings["T_lower"], settings["T_upper"]),
        (0.0, settings["alpha_upper"]),
    ]

    result_global = differential_evolution(
        global_objective,
        bounds=bounds,
        seed=123,
        popsize=settings["global_popsize"],
        maxiter=settings["global_maxiter"],
        tol=settings["global_tol"],
        polish=False,
        updating="immediate",
        workers=1,
    )

    def local_objective(x):
        eval_counter["Local"] += 1

        T = float(x[0])
        alpha = float(x[1])

        start = time.perf_counter()
        value = cost_rate(T, alpha * T)
        elapsed = time.perf_counter() - start

        add_log("Local", T, alpha, value, elapsed)
        return value

    result_local = minimize(
        local_objective,
        x0=result_global.x,
        method="Nelder-Mead",
        options={
            "xatol": 1e-6,
            "fatol": 1e-8,
            "maxiter": settings["local_maxiter"],
            "disp": False,
        },
    )

    T_star = float(result_local.x[0])
    alpha_star = float(result_local.x[1])
    tau_star = alpha_star * T_star
    C_star = float(cost_rate(T_star, tau_star))

    return {
        "T_star": T_star,
        "tau_star": tau_star,
        "alpha_star": alpha_star,
        "C_star": C_star,
        "result_global": result_global,
        "result_local": result_local,
        "log": pd.DataFrame(log_rows),
    }


# ============================================================
# USER INTERFACE
# ============================================================

st.header("General parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    ci = st.number_input(
        "Scheduled intervention cost (Ci)",
        min_value=0.0,
        value=500.0,
        step=50.0,
        help="Fixed cost associated with a scheduled inspection/intervention.",
    )

with col2:
    co = st.number_input(
        "Opportunistic intervention cost (Co)",
        min_value=0.0,
        value=300.0,
        step=50.0,
        help="Fixed cost associated with an opportunistic inspection/intervention.",
    )

with col3:
    cf = st.number_input(
        "Base corrective failure cost (Cf)",
        min_value=0.0,
        value=1500.0,
        step=100.0,
        help="General system-level cost caused by a failure event.",
    )

with col4:
    mu = st.number_input(
        "Opportunity arrival rate (μ)",
        min_value=1e-8,
        value=0.001,
        step=0.0001,
        format="%.6f",
        help="Rate of the Homogeneous Poisson Process that generates opportunities.",
    )

st.divider()

st.header("Component structure")

n_types = st.number_input(
    "Number of distinct component types",
    min_value=1,
    max_value=20,
    value=4,
    step=1,
    help=(
        "Use component types to avoid repeated data entry. "
        "For example, if the system has 5 identical bearings, enter one component type "
        "with quantity equal to 5."
    ),
)

quantities = []
lambda_x = []
beta_h = []
eta_h = []
cef = []

default_lambda = [0.0015, 0.0010, 0.0007, 0.0005]
default_beta = [2.0, 2.5, 3.0, 2.2]
default_eta = [180.0, 250.0, 350.0, 450.0]
default_cef = [500.0, 800.0, 1000.0, 1200.0]

st.markdown("### Reliability and cost parameters by component type")

for i in range(int(n_types)):
    with st.expander(f"Component type {i + 1}", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            q = st.number_input(
                f"Quantity of type {i + 1}",
                min_value=1,
                value=1,
                step=1,
                key=f"quantity_{i}",
                help="Number of identical components represented by this type.",
            )

        with c2:
            lam = st.number_input(
                f"λ for X, type {i + 1}",
                min_value=1e-8,
                value=default_lambda[i] if i < len(default_lambda) else 0.001,
                step=0.0001,
                format="%.6f",
                key=f"lambda_{i}",
                help="Rate of the exponential time-to-defect distribution.",
            )

        with c3:
            beta = st.number_input(
                f"Weibull β, type {i + 1}",
                min_value=0.1,
                value=default_beta[i] if i < len(default_beta) else 2.0,
                step=0.1,
                format="%.4f",
                key=f"beta_{i}",
                help="Shape parameter of the Weibull delay-time distribution.",
            )

        with c4:
            eta = st.number_input(
                f"Weibull η, type {i + 1}",
                min_value=1e-8,
                value=default_eta[i] if i < len(default_eta) else 250.0,
                step=10.0,
                format="%.4f",
                key=f"eta_{i}",
                help="Scale parameter of the Weibull delay-time distribution.",
            )

        with c5:
            c_extra = st.number_input(
                f"Extra failure cost, type {i + 1}",
                min_value=0.0,
                value=default_cef[i] if i < len(default_cef) else 500.0,
                step=100.0,
                format="%.4f",
                key=f"cef_{i}",
                help="Additional cost associated with a failure caused by this component type.",
            )

        quantities.append(q)
        lambda_x.append(lam)
        beta_h.append(beta)
        eta_h.append(eta)
        cef.append(c_extra)

input_df = pd.DataFrame(
    {
        "Component type": [f"Type {i + 1}" for i in range(int(n_types))],
        "Quantity": quantities,
        "lambda_X": lambda_x,
        "beta_H": beta_h,
        "eta_H": eta_h,
        "Extra failure cost": cef,
    }
)

st.markdown("### Input summary")
st.dataframe(input_df, use_container_width=True, hide_index=True)

st.divider()

st.info(
    "Numerical and optimization settings are selected automatically from the input data. "
    "This keeps the interface simple while adapting the grid and search range to the reliability scale."
)

run_button = st.button("Run optimization", type="primary")


# ============================================================
# MODEL EXECUTION
# ============================================================

if run_button:
    total_start = time.perf_counter()

    quantities = np.asarray(quantities, dtype=float)
    lambda_x = np.asarray(lambda_x, dtype=float)
    beta_h = np.asarray(beta_h, dtype=float)
    eta_h = np.asarray(eta_h, dtype=float)
    cef = np.asarray(cef, dtype=float)

    settings = automatic_settings(
        quantities=quantities,
        lambda_x=lambda_x,
        beta_h=beta_h,
        eta_h=eta_h,
    )

    t_max = settings["t_max"]

    st.subheader("Automatic numerical settings")

    settings_df = pd.DataFrame(
        [
            {"Setting": "Characteristic system scale", "Value": settings["system_scale"]},
            {"Setting": "Total number of components", "Value": settings["total_components"]},
            {"Setting": "Lower bound for T", "Value": settings["T_lower"]},
            {"Setting": "Upper bound for T", "Value": settings["T_upper"]},
            {"Setting": "Grid horizon", "Value": settings["t_max"]},
            {"Setting": "Time step dt", "Value": settings["dt"]},
            {"Setting": "Quadrature points", "Value": settings["n_quad"]},
            {"Setting": "Global search iterations", "Value": settings["global_maxiter"]},
            {"Setting": "Global population size", "Value": settings["global_popsize"]},
        ]
    )

    st.dataframe(settings_df, use_container_width=True, hide_index=True)

    st.subheader("Model construction")

    model_status = st.status(
        "Building reliability and renewal curves...",
        expanded=True,
    )

    with model_status:
        st.write("Computing component failure distributions.")
        st.write("Computing system first-failure density.")
        st.write("Computing renewal-based expected failures by component type.")

        t, mj_grid, info = build_model(
            quantities=quantities,
            lambda_x=lambda_x,
            beta_h=beta_h,
            eta_h=eta_h,
            dt=settings["dt"],
            t_max=t_max,
            max_renewal_terms=settings["max_renewal_terms"],
            renewal_tol=settings["renewal_tol"],
        )

        st.write("Model construction completed.")

    model_status.update(
        label="Reliability and renewal curves built successfully.",
        state="complete",
        expanded=False,
    )

    col_a, col_b, col_c, col_d = st.columns(4)

    col_a.metric("Grid points", f"{info['n_grid']}")
    col_b.metric("Build time", f"{info['build_time']:.2f} s")
    col_c.metric("Renewal stop term", f"{info['renewal_stop_term']}")
    col_d.metric("System first-failure mass", f"{info['system_first_failure_mass']:.6f}")

    mass_df = pd.DataFrame(
        {
            "Component type": [f"Type {i + 1}" for i in range(int(n_types))],
            "Integral of f_Z over grid": info["component_masses"],
        }
    )

    st.markdown("#### Probability mass checks")
    st.dataframe(mass_df, use_container_width=True, hide_index=True)

    with st.expander("Renewal calculation log", expanded=False):
        for msg in info["progress_messages"]:
            st.text(msg)

    cost_rate, ec_i, ev_i, ec_o, ev_o = make_cost_functions(
        t=t,
        mj_grid=mj_grid,
        cef=cef,
        ci=float(ci),
        co=float(co),
        cf=float(cf),
        mu=float(mu),
        n_quad=settings["n_quad"],
    )

    st.subheader("Optimization progress")
    st.info("The table below updates while the optimizer tests candidate solutions.")
    progress_container = st.empty()

    opt_status = st.status(
        "Running optimization...",
        expanded=True,
    )

    with opt_status:
        st.write("Running global search.")
        st.write("Running local refinement after the global search.")

        results = optimize_policy(
            cost_rate=cost_rate,
            settings=settings,
            progress_container=progress_container,
        )

        st.write("Optimization completed.")

    opt_status.update(
        label="Optimization completed successfully.",
        state="complete",
        expanded=False,
    )

    T_star = results["T_star"]
    tau_star = results["tau_star"]
    alpha_star = results["alpha_star"]
    C_star = results["C_star"]

    st.subheader("Optimal policy")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Optimal T*", f"{T_star:.6f}")
    m2.metric("Optimal τ*", f"{tau_star:.6f}")
    m3.metric("τ*/T*", f"{alpha_star:.6f}")
    m4.metric("Cost-rate C∞", f"{C_star:.8f}")

    st.markdown("#### Cycle quantities")

    pi_i = prob_no_opportunity(tau_star, float(mu))
    pi_o = 1.0 - pi_i

    cycle_df = pd.DataFrame(
        {
            "Quantity": [
                "EC_i",
                "EV_i",
                "EC_o",
                "EV_o",
                "pi_i",
                "pi_o",
            ],
            "Value": [
                ec_i(T_star, tau_star),
                ev_i(T_star, tau_star),
                ec_o(T_star, tau_star),
                ev_o(T_star, tau_star),
                pi_i,
                pi_o,
            ],
        }
    )

    st.dataframe(cycle_df, use_container_width=True, hide_index=True)

    st.markdown("#### Tested solutions")
    st.dataframe(results["log"], use_container_width=True, hide_index=True)

    csv_log = results["log"].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download tested solutions as CSV",
        data=csv_log,
        file_name="optimization_tested_solutions.csv",
        mime="text/csv",
    )

    # Periodic policy comparison
    st.subheader("Periodic policy comparison")

    def periodic_objective(x):
        return cost_rate(float(x[0]), 0.0)

    periodic_status = st.status(
        "Optimizing periodic policy...",
        expanded=False,
    )

    with periodic_status:
        result_periodic = minimize(
            periodic_objective,
            x0=np.array([T_star]),
            bounds=[(settings["T_lower"], settings["T_upper"])],
            method="L-BFGS-B",
        )

    periodic_status.update(
        label="Periodic policy optimization completed.",
        state="complete",
        expanded=False,
    )

    T_periodic = float(result_periodic.x[0])
    C_periodic = float(result_periodic.fun)
    improvement = 100.0 * (C_periodic - C_star) / C_periodic

    p1, p2, p3 = st.columns(3)

    p1.metric("Periodic T", f"{T_periodic:.6f}")
    p2.metric("Periodic cost-rate", f"{C_periodic:.8f}")
    p3.metric("Improvement over periodic", f"{improvement:.4f}%")

    # Numerical validation
    st.subheader("Numerical validation")

    validation_status = st.status(
        "Running numerical validation...",
        expanded=False,
    )

    validation_rows = []

    with validation_status:
        st.write("Checking sensitivity to quadrature points.")

        for nq in [50, 100, 150, 200]:
            cr_tmp, _, _, _, _ = make_cost_functions(
                t=t,
                mj_grid=mj_grid,
                cef=cef,
                ci=float(ci),
                co=float(co),
                cf=float(cf),
                mu=float(mu),
                n_quad=int(nq),
            )

            C_test = cr_tmp(T_star, tau_star)

            validation_rows.append(
                {
                    "Test": "n_quad sensitivity",
                    "Setting": f"n_quad = {nq}",
                    "Cost-rate": C_test,
                    "Difference from base": C_test - C_star,
                }
            )

        st.write("Checking sensitivity to the time step.")

        for dt_test in [0.5, 0.25, 0.1]:
            t_test, mj_test, info_test = build_model(
                quantities=quantities,
                lambda_x=lambda_x,
                beta_h=beta_h,
                eta_h=eta_h,
                dt=float(dt_test),
                t_max=t_max,
                max_renewal_terms=settings["max_renewal_terms"],
                renewal_tol=settings["renewal_tol"],
            )

            cr_test, _, _, _, _ = make_cost_functions(
                t=t_test,
                mj_grid=mj_test,
                cef=cef,
                ci=float(ci),
                co=float(co),
                cf=float(cf),
                mu=float(mu),
                n_quad=settings["n_quad"],
            )

            C_test = cr_test(T_star, tau_star)

            validation_rows.append(
                {
                    "Test": "dt sensitivity",
                    "Setting": f"dt = {dt_test}",
                    "Cost-rate": C_test,
                    "Difference from base": C_test - C_star,
                }
            )

        st.write("Checking local stability around the optimum.")

        perturbations = [
            (-0.02, 0.00),
            (0.02, 0.00),
            (0.00, -0.02),
            (0.00, 0.02),
            (-0.02, -0.02),
            (0.02, 0.02),
        ]

        for dT_rel, dtau_rel in perturbations:
            T_test = T_star * (1.0 + dT_rel)
            tau_test = tau_star * (1.0 + dtau_rel)

            if tau_test >= T_test:
                continue

            C_test = cost_rate(T_test, tau_test)

            validation_rows.append(
                {
                    "Test": "local stability",
                    "Setting": f"T {dT_rel:+.0%}, tau {dtau_rel:+.0%}",
                    "Cost-rate": C_test,
                    "Difference from base": C_test - C_star,
                }
            )

    validation_status.update(
        label="Numerical validation completed.",
        state="complete",
        expanded=False,
    )

    validation_df = pd.DataFrame(validation_rows)
    st.dataframe(validation_df, use_container_width=True, hide_index=True)

    # Reference point check
    st.subheader("Reference point check")

    ref_col1, ref_col2, ref_col3 = st.columns(3)

    with ref_col1:
        T_ref = st.number_input(
            "Reference T",
            min_value=0.0,
            value=230.20,
            step=1.0,
            format="%.4f",
        )

    with ref_col2:
        tau_ref = st.number_input(
            "Reference tau",
            min_value=0.0,
            value=118.99,
            step=1.0,
            format="%.4f",
        )

    with ref_col3:
        C_ref = st.number_input(
            "Reference cost-rate",
            min_value=0.0,
            value=3.6337,
            step=0.01,
            format="%.6f",
        )

    if tau_ref < T_ref:
        C_at_ref = cost_rate(float(T_ref), float(tau_ref))

        r1, r2 = st.columns(2)
        r1.metric("Model cost-rate at reference point", f"{C_at_ref:.8f}")
        r2.metric("Difference from reference", f"{C_at_ref - C_ref:+.8f}")
    else:
        st.warning("Reference tau must be smaller than reference T.")

    total_elapsed = time.perf_counter() - total_start
    st.success(f"Run completed in {total_elapsed:.2f} seconds.")
