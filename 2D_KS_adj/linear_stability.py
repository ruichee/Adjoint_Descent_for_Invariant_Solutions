import numpy as np
import matplotlib.pyplot as plt
from dealiase import dealiase
from scipy.sparse.linalg import LinearOperator, eigs
from get_R import get_R
# ==========================================
# 1. SETUP AND GRID PARAMETERS
# ==========================================
# TODO: Replace with your actual domain parameters
nx, ny = 64, 64
Lx, Ly = 20, 20  # Remember: If your domain is [0, 2*L_half], Lx here is the FULL length

# TODO: Load your fixed point here
u_fixed = np.loadtxt(r"2D_KS_adj\fixed_points\output_u.dat", delimiter=" ")


def compute_leading_eigenvalue(u_fixed, nx, ny):
    N = nx * ny
    u_flat = u_fixed.flatten()
    
    # 1. Pre-compute the baseline RHS
    u_fixed_2d = u_fixed.reshape((nx, ny))
    R_fixed_flat = get_R(0, u_fixed_2d).flatten()
    
    # --- FIX 1: Increase epsilon to survive the noise floor ---
    epsilon = 1e-5 
    
    def jacobian_action(v_1d):
        u_perturbed_1d = u_flat + epsilon * v_1d
        u_perturbed_2d = u_perturbed_1d.reshape((nx, ny))
        R_perturbed_flat = get_R(0, u_perturbed_2d).flatten()
        
        Jv = (R_perturbed_flat - R_fixed_flat) / epsilon
        return Jv

    # Diagnostic: Test the Jacobian action once to make sure it's alive
    test_vec = np.random.rand(N)
    test_out = jacobian_action(test_vec)
    if np.allclose(test_out, 0):
        print("WARNING: Jacobian action returned all zeros. Epsilon might still be too small.")

    # 3. Create the SciPy Linear Operator
    J_op = LinearOperator((N, N), matvec=jacobian_action)
    
    print("Computing leading eigenvalue with expanded subspace...")
    
    # --- FIX 2: Expand subspace (ncv) and relax tolerance ---
    # ncv=30 gives it 30 vectors to untangle the clustered eigenvalues
    # tol=1e-4 prevents it from obsessing over machine-precision accuracy
    eigenvalues, eigenvectors = eigs(J_op, k=1, which='LR', ncv=30, tol=1e-5)
    
    leading_eval = np.real(eigenvalues[0])
    print(f"Predicted Slope (Leading Eigenvalue): {leading_eval:.6f}")
    
    dominant_mode_2d = np.real(eigenvectors[:, 0]).reshape((nx, ny))
    return leading_eval, dominant_mode_2d

# --- Usage ---
# Assuming you have loaded u_fixed
slope, dominant_mode = compute_leading_eigenvalue(u_fixed, 64, 64)

predicted_slope = slope  # The theoretical eigenvalue you found

# ==========================================
# 2. SPECTRAL GRID & OPERATORS
# ==========================================
# Wave numbers
kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
KX, KY = np.meshgrid(kx, ky, indexing='ij')

# The Linear Operator (L = q^2 - q^4) for the 2D Kuramoto-Sivashinsky Eq.
q2 = KX**2 + KY**2
L_op = q2 - q2**2

# Dealiasing Mask (The 2/3 Rule to prevent aliasing shock)
kmax_x, kmax_y = np.max(kx), np.max(ky)
dealias_mask = (np.abs(KX) < (2.0/3.0)*kmax_x) & (np.abs(KY) < (2.0/3.0)*kmax_y)

# ==========================================
# 3. ETDRK4 ENGINE (Kassam-Trefethen)
# ==========================================
def setup_etdrk4(L, dt):
    """Pre-computes ETDRK4 coefficients using complex contour integration."""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    
    # Contour integration for stability at L=0
    M = 32 
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[..., np.newaxis] + r
    
    Q  = dt * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1))
    f3 = dt * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1))
    
    return E, E2, Q, f1, f2, f3

def compute_N(u_hat):
    """
    Exact 1:1 mapping of your get_R.py non-linear components
    """
    # 1. Match your exact dealiasing of the input state
    u_f = dealiase(u_hat) 
    
    # 2. Match your exact derivative calculations
    u_x_f = 1j * KX * u_f
    u_y_f = 1j * KY * u_f
    
    # 3. Match your exact complex ifft2 (without np.real cast)
    u_x = np.real(np.fft.ifft2(u_x_f))
    u_y = np.real(np.fft.ifft2(u_y_f))
    
    # 4. Match your exact nonlinear term
    u_sq_terms = -0.5 * (u_x*u_x + u_y*u_y)
    
    # 5. Back to Fourier and apply your exact dealiase
    N_hat = np.fft.fft2(u_sq_terms)
    N_hat = dealiase(N_hat)
    
    # 6. Match your exact mean-flow masking
    mask = (KX == 0) & (KY == 0)
    N_hat = np.where(mask, 0, N_hat)
    
    return N_hat

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

# ==========================================
# 4. INITIALIZATION & NOISE INJECTION
# ==========================================
dt = 0.01      # ETDRK4 can comfortably take large steps
T_end = 200.0  
num_steps = int(T_end / dt)

# artificial injection of noise if time step is too large to inject it numerically
'''# Generate structurally safe noise (Low-frequency, strictly zero-mean)
np.random.seed(42)
raw_noise = np.random.randn(nx, ny)
noise_hat = np.fft.fft2(raw_noise)

# Filter out high frequencies to avoid hyperdiffusion shock
noise_hat[(np.abs(KX) > 3) | (np.abs(KY) > 3)] = 0.0

# STRICTLY enforce zero spatial mean
noise_hat[0, 0] = 0.0 

smooth_noise = np.real(np.fft.ifft2(noise_hat))
epsilon = 1e-12
smooth_noise = (smooth_noise / np.linalg.norm(smooth_noise)) * epsilon'''

# Create the perturbed starting state in Fourier space
u_fixed_2d = u_fixed.reshape((nx, ny))
u_imperfect = u_fixed_2d 
'''+ smooth_noise'''

# Initial condition for the solver
u_hat = np.fft.fft2(u_imperfect) * dealias_mask
u_hat[0, 0] = 0.0

# Pre-compute solver matrices
print("Initializing ETDRK4 Coefficients...")
E, E2, Q, f1, f2, f3 = setup_etdrk4(L_op, dt)

# Project the reference state exactly like the solver does
u_fixed_hat = np.fft.fft2(u_fixed_2d) * dealias_mask
u_fixed_hat[0, 0] = 0.0

# Create the true physical baseline for our distance metric
u_fixed_baseline = np.real(np.fft.ifft2(u_fixed_hat))

# ==========================================
# 5. MAIN TIME-STEPPING LOOP
# ==========================================
time_record = []
norm_record = []

print(f"Running DNS for {num_steps} steps...")
for step in range(num_steps):
    # Take step
    u_hat = etdrk4_step(u_hat, E, E2, Q, f1, f2, f3)
    
    # Record data every 10 steps to save memory
    if step % 10 == 0:
        t = step * dt
        # Euclidean norm of the physical difference
        u_physical = np.real(np.fft.ifft2(u_hat))
        dist = np.linalg.norm(u_physical - u_fixed_baseline)
        
        time_record.append(t)
        norm_record.append(dist)

# ==========================================
# 6. PLOTTING AND VALIDATION
# ==========================================
time_record = np.array(time_record)
norm_record = np.array(norm_record)

# Construct theoretical line
# Pick an anchor point right as it breaks out of the noise floor (e.g., t=30)
t_anchor = 30.0
idx_anchor = np.searchsorted(time_record, t_anchor)
y_anchor = norm_record[idx_anchor]

theoretical_line = y_anchor * np.exp(predicted_slope * (time_record - t_anchor))

plt.figure(figsize=(8, 6))
plt.semilogy(time_record, norm_record, '-b', linewidth=2.5, label='2D KSE ETDRK4 DNS')
plt.semilogy(time_record, theoretical_line, '--r', linewidth=2, 
             label=f'Linear stability analysis: slope = {predicted_slope:.4f}')

plt.ylim(1e-13, 1e4) 
plt.xlim(-10, 210)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('||U(t) - U(t_0)||_2', fontsize=12)
plt.title('L2-norm of difference over time', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower right', fontsize=11)
plt.show()