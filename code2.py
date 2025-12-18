import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
def calculate_vibrator_pattern(l_over_lambda, theta_degrees):
    theta = np.radians(theta_degrees)
    l = l_over_lambda
    k_l = 2 * np.pi * l
    sin_theta = np.sin(theta)
    sin_theta[sin_theta == 0] = 1e-10
    F_theta = (np.cos(k_l * np.cos(theta)) - np.cos(k_l)) / sin_theta
    F_theta_norm = np.abs(F_theta) / np.max(np.abs(F_theta))
    
    phi = np.linspace(0, 2*np.pi, 360)
    Theta, Phi = np.meshgrid(theta, phi)
    sin_theta_grid = np.sin(Theta)
    sin_theta_grid[sin_theta_grid == 0] = 1e-10
    F_theta_grid = (np.cos(k_l * np.cos(Theta)) - np.cos(k_l)) / sin_theta_grid
    integrand = np.abs(F_theta_grid)**2 * np.sin(Theta)
    integral = np.trapz(np.trapz(integrand, phi, axis=0), theta)
    D_theta = (4 * np.pi * np.abs(F_theta)**2) / integral
    
    return D_theta, F_theta_norm
def load_cst_data(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        data_lines = []
        for line in lines:
            if line.strip() and not line.startswith('---') and not line.startswith('Theta'):
                data_lines.append(line)
        theta_cst = []
        dir_cst = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    theta = float(parts[0])
                    directivity = float(parts[2])
                    theta_cst.append(theta)
                    dir_cst.append(directivity)
                except ValueError:
                    continue
        
        return np.array(theta_cst), np.array(dir_cst)
    except FileNotFoundError:
        return np.array([]), np.array([])
def create_plots_with_cst_comparison(theta_degrees, F_norm, D_theta, D_theta_dB, 
                                    cst_theta, cst_D, params):
    theta_degrees_full = np.concatenate([theta_degrees, 360 - theta_degrees[::-1]])
    F_norm_full = np.concatenate([F_norm, F_norm[::-1]])
    D_theta_full = np.concatenate([D_theta, D_theta[::-1]])
    D_theta_dB_full = np.concatenate([D_theta_dB, D_theta_dB[::-1]])
    theta_rad_full = np.radians(theta_degrees_full)
    if len(cst_theta) > 0:
        cst_mask = (cst_theta <= 180)
        cst_theta_180 = cst_theta[cst_mask]
        cst_D_180 = cst_D[cst_mask]
        cst_F_norm = cst_D_180 / np.max(cst_D_180)
        cst_theta_full = np.concatenate([cst_theta_180, 360 - cst_theta_180[::-1]])
        cst_F_norm_full = np.concatenate([cst_F_norm, cst_F_norm[::-1]])
        cst_D_full = np.concatenate([cst_D_180, cst_D_180[::-1]])
        cst_D_dB_full = 10 * np.log10(cst_D_full)
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(theta_degrees, F_norm, 'b-', linewidth=2, label='Теория')
    if len(cst_theta) > 0:
        unique_indices = np.arange(0, len(cst_theta_180), 5)
        ax1.plot(cst_theta_180[unique_indices], cst_F_norm[unique_indices], 
                'ro', markersize=4, alpha=0.7, label='CST')
    ax1.set_xlabel('Угол θ, градусы')
    ax1.set_ylabel('Нормированная амплитуда поля, |F(θ)|')
    ax1.set_title('Нормированная ДН (линейная шкала)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax2 = plt.subplot(2, 2, 2, projection='polar')
    ax2.plot(theta_rad_full, F_norm_full, 'b-', linewidth=2, alpha=0.7, label='Теория')
    if len(cst_theta) > 0:
        ax2.plot(np.radians(cst_theta_full), cst_F_norm_full, 'r--', linewidth=1.5, alpha=0.7, label='CST')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title('Нормированная ДН (полярные координаты)', pad=20)
    ax2.grid(True)
    ax2.set_rmax(1.1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(theta_degrees, D_theta_dB, 'g-', linewidth=2, label='Теория')
    ax3.axhline(y=params['D_max_dB_theory'], color='b', linestyle='--', alpha=0.5, 
               label=f'Теория: Dmax = {params["D_max_dB_theory"]:.4f} дБ')
    if len(cst_theta) > 0:
        cst_D_dB_180 = 10 * np.log10(cst_D_180)
        unique_indices = np.arange(0, len(cst_theta_180), 5)
        ax3.plot(cst_theta_180[unique_indices], cst_D_dB_180[unique_indices], 
                'ms', markersize=4, alpha=0.7, label='CST')
        ax3.axhline(y=params['D_max_dB_cst'], color='r', linestyle='--', alpha=0.5,
                   label=f'CST: Dmax = {params["D_max_dB_cst"]:.4f} дБ')
    ax3.set_xlabel('Угол θ, градусы')
    ax3.set_ylabel('КНД, D(θ), дБ')
    ax3.set_title('Зависимость КНД от угла (логарифмическая шкала)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, 180)
    ax3.set_ylim(-50, 5)
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    D_theta_dB_nonneg = D_theta_dB_full - np.min(D_theta_dB_full)
    ax4.plot(theta_rad_full, D_theta_dB_nonneg, 'b-', linewidth=2, alpha=0.7, label='Теория')
    if len(cst_theta) > 0:
        cst_D_dB_nonneg = cst_D_dB_full - np.min(cst_D_dB_full)
        ax4.plot(np.radians(cst_theta_full), cst_D_dB_nonneg, 'r--', linewidth=1.5, alpha=0.7, label='CST')
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_title('КНД в дБ (полярные координаты)', pad=20)
    ax4.grid(True)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.suptitle(f'Диаграмма направленности симметричного вибратора\n'
                 f'2l/λ = {params["2l_over_lambda"]:.3f}, f = {params["f_ghz"]} ГГц\n'
                 f'Сравнение: Теория vs CST Studio Suite', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()
def main():
    f = 1.0e9
    c = 3.0e8
    l_over_lambda = 0.01 / 2
    theta_degrees = np.linspace(0, 180, 361)
    theta_degrees = theta_degrees[1:-1]
    D_theta, F_norm = calculate_vibrator_pattern(l_over_lambda, theta_degrees)
    D_max_theory = np.max(D_theta)
    D_max_dB_theory = 10 * np.log10(D_max_theory)
    D_theta_dB = 10 * np.log10(D_theta)
    cst_filename = "dipol132.txt"
    cst_theta, cst_D = load_cst_data(cst_filename) 
    if len(cst_theta) > 0:
        cst_mask = (cst_theta <= 180)
        D_max_cst = np.max(cst_D[cst_mask])
        D_max_dB_cst = 10 * np.log10(D_max_cst)
    else:
        cst_theta = np.array([])
        cst_D = np.array([])
        D_max_cst = 0
        D_max_dB_cst = 0
    params = {
        'l_over_lambda': l_over_lambda,
        '2l_over_lambda': 2*l_over_lambda,
        'f_ghz': f/1e9,
        'D_max_theory': D_max_theory,
        'D_max_dB_theory': D_max_dB_theory,
        'D_max_cst': D_max_cst,
        'D_max_dB_cst': D_max_dB_cst
    }
    create_plots_with_cst_comparison(theta_degrees, F_norm, D_theta, 
                                     D_theta_dB, cst_theta, cst_D, params)
if __name__ == "__main__":
    main()