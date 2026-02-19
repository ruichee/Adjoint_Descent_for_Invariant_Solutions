def etdrk4_step(v, E, E2, Q, f1, f2, f3):
    """Executes a single ETDRK4 time step."""
    Nv = compute_N(v)
    a = E2 * v + Q * Nv
    
    Na = compute_N(a)
    b = E2 * v + Q * Na
    
    Nb = compute_N(b)
    c = E2 * a + Q * (2 * Nb - Nv)
    
    Nc = compute_N(c)
    v_new = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    
    # Double check mean-flow is suppressed
    v_new[0, 0] = 0.0 + 0.0j
    return v_new