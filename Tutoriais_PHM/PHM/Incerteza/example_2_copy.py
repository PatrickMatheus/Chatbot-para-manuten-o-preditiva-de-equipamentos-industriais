import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

t_data = np.array([0, 1, 2, 3, 4])
z_data = np.array([6.99, 2.28, 1.91, 11.94, 14.60])
L = 1
h1_known = 5


H = np.column_stack((L * t_data**2, t_data**3))
z_tilde = z_data - h1_known
h_hat = np.linalg.inv(H.T @ H) @ H.T @ z_tilde

residuals = z_tilde - H @ h_hat
sigma2 = np.sum(residuals**2) / (len(z_data) - len(h_hat))
Cov_h = sigma2 * np.linalg.inv(H.T @ H)

print("Estimativa dos parâmetros:", h_hat)
print("Covariância dos parâmetros:\n", Cov_h)


n_samples = 5000
dof = len(z_data) - len(h_hat)

t_samples = t.rvs(df=dof, size=(n_samples, 2))
L_chol = np.linalg.cholesky(Cov_h)
h_samples = h_hat + t_samples @ L_chol.T

t_future = np.arange(4, 16)
z_future_samples = np.array([h1_known + h_s[0]*L*t_future**2 + h_s[1]*t_future**3 for h_s in h_samples])

z_median = np.median(z_future_samples, axis=0)
z_ci90 = np.percentile(z_future_samples, [5,95], axis=0)

theta_true = np.array([0.2, 0.1])
X_new = np.column_stack((L * t_future**2, t_future**3))
z_true = h1_known + X_new @ theta_true

plt.figure(figsize=(10,6))

for sample in z_future_samples[::50]:  # plotar uma em cada 50 para não poluir
    plt.plot(t_future, sample, color='gray', alpha=0.1)
plt.plot(t_future, z_median, 'b--', label='Mediana')
plt.fill_between(t_future, z_ci90[0], z_ci90[1], color='lightgray', alpha=0.5, label='90% CI')
plt.plot(t_future, z_true, 'r', linewidth=2, label='Curva verdadeira')
plt.scatter(t_data, z_data, color='black', label='Dados ruidosos')
plt.axhline(150, color='green', linestyle='--', label='Limite RUL')
plt.title("Previsão da degradação futura")
plt.xlabel("t")
plt.ylabel("z(t)")
plt.legend()
plt.show()


threshold = 150
t_current = 4
RUL_samples = []

for sample in z_future_samples:
    above = np.where(sample >= threshold)[0]
    if len(above) == 0:
        continue
    elif above[0] == 0:
        RUL_samples.append(0)
    else:
        i1 = above[0]
        i0 = i1 - 1
        z0, z1 = sample[i0], sample[i1]
        t0, t1 = t_future[i0], t_future[i1]
        t_fail = t0 + (threshold - z0) * (t1 - t0) / (z1 - z0)
        RUL_samples.append(t_fail - t_current)

RUL_samples = np.array(RUL_samples)
RUL_median = np.median(RUL_samples)
RUL_ci90 = np.percentile(RUL_samples, [5,95])

print("RUL mediano:", RUL_median)
print("RUL 90% CI:", RUL_ci90)

plt.figure(figsize=(8,5))
plt.hist(RUL_samples, bins=30, alpha=0.7, color='orange')
plt.axvline(RUL_median, color='blue', linestyle='--', label='Mediana RUL')
plt.axvline(RUL_ci90[0], color='red', linestyle=':', label='5% CI')
plt.axvline(RUL_ci90[1], color='red', linestyle=':', label='95% CI')

above_true = np.where(z_true >= threshold)[0]
if len(above_true) == 0:
    RUL_true = np.nan
elif above_true[0] == 0:
    RUL_true = 0
else:
    i1 = above_true[0]
    i0 = i1 - 1
    z0, z1 = z_true[i0], z_true[i1]
    t0, t1 = t_future[i0], t_future[i1]
    RUL_true = t0 + (threshold - z0) * (t1 - t0) / (z1 - z0) - t_current

plt.axvline(RUL_true, color='green', linestyle='--', linewidth=2, label='RUL verdadeira')
plt.title("Distribuição do RUL")
plt.xlabel("RUL")
plt.ylabel("Frequência")
plt.legend()
plt.show()
