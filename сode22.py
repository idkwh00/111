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
        print(f"Ошибка: Файл {filename} не найден!")
        return np.array([]), np.array([])

def create_plots_with_cst_comparison(theta_degrees, F_norm, D_theta, D_theta_dB, 
                                    cst_theta, cst_D, params):
    theta_degrees_full = np.concatenate([theta_degrees, 360 - theta_degrees[::-1]])
    F_norm_full = np.concatenate([F_norm, F_norm[::-1]])
    D_theta_full = np.concatenate([D_theta, D_theta[::-1]])
    D_theta_dB_full = np.concatenate([D_theta_dB, D_theta_dB[::-1]])
    theta_rad_full = np.radians(theta_degrees_full)
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
    unique_indices = np.arange(0, len(cst_theta_180), 5)
    if len(unique_indices) > 0:
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
    if len(cst_theta_full) > 0:
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
    if len(cst_theta_180) > 0:
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
    if len(cst_D_dB_full) > 0:
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
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    if len(cst_theta_180) > 0:
        cst_D_interp = np.interp(theta_degrees, cst_theta_180, cst_D_180)
        valid_mask = (D_theta > 1e-6) & (cst_D_interp > 1e-6)
        if np.any(valid_mask):
            rel_error = np.abs(D_theta[valid_mask] - cst_D_interp[valid_mask]) / D_theta[valid_mask] * 100
            ax.plot(theta_degrees[valid_mask], rel_error, 'r-', linewidth=2)
            ax.fill_between(theta_degrees[valid_mask], 0, rel_error, alpha=0.3, color='red')
            mean_error = np.mean(rel_error)
            ax.axhline(y=mean_error, color='blue', linestyle='--', 
                      label=f'Средняя ошибка: {mean_error:.2f}%')
            ax.set_xlabel('Угол θ, градусы')
            ax.set_ylabel('Относительная ошибка, %')
            ax.set_title('Относительная ошибка между теорией и CST')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 180)
            ax.set_ylim(0, max(rel_error) * 1.1)
            ax.legend()
            plt.tight_layout()
            plt.show()
            return mean_error 
    return 0
def main():
    f = 1.0e9
    c = 3.0e8
    l_over_lambda = 0.01 / 2 
    print("=" * 80)
    print("ДИАГРАММА НАПРАВЛЕННОСТИ СИММЕТРИЧНОГО ВИБРАТОРА")
    print("Сравнение теоретических расчетов с CST Studio Suite")
    print("=" * 80)
    print(f"Частота: f = {f/1e9} ГГц")
    print(f"Отношение длины вибратора к длине волны: 2l/λ = {2*l_over_lambda:.3f}")
    print(f"Длина плеча: l/λ = {l_over_lambda:.4f}")
    print(f"Длина волны: λ = {c/f:.3f} м")
    print(f"Длина вибратора: 2l = {2*l_over_lambda * c/f:.6f} м")
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
        print("\n⚠ Файл CST не найден. Будут показаны только теоретические графики.")
        cst_theta = np.array([])
        cst_D = np.array([])
        D_max_cst = 0
        D_max_dB_cst = 0
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ РАСЧЕТА:")
    print("=" * 80)
    print(f"Теоретический максимальный КНД:")
    print(f"  - в разах: {D_max_theory:.6f}")
    print(f"  - в децибелах: {D_max_dB_theory:.6f} дБ")
    if len(cst_theta) > 0:
        print(f"\nCST максимальный КНД:")
        print(f"  - в разах: {D_max_cst:.6f}")
        print(f"  - в децибелах: {D_max_dB_cst:.6f} дБ")
        print(f"\nРазница в максимальном КНД: {abs(D_max_theory - D_max_cst):.6f} раз")
        print(f"Разница в дБ: {abs(D_max_dB_theory - D_max_dB_cst):.6f} дБ")
    max_angle_idx = np.argmax(D_theta)
    max_angle = theta_degrees[max_angle_idx]
    print(f"\nУгол максимума излучения (теория): {max_angle:.1f}°")
    D_theoretical = 1.5
    D_theoretical_dB = 10 * np.log10(D_theoretical)
    print(f"\nДля сравнения, для элементарного диполя (очень короткого вибратора):")
    print(f"  - Dmax: {D_theoretical:.6f} раз ({D_theoretical_dB:.6f} дБ)")
    params = {
        'l_over_lambda': l_over_lambda,
        '2l_over_lambda': 2*l_over_lambda,
        'f_ghz': f/1e9,
        'D_max_theory': D_max_theory,
        'D_max_dB_theory': D_max_dB_theory,
        'D_max_cst': D_max_cst,
        'D_max_dB_cst': D_max_dB_cst,
        'D_theoretical': D_theoretical
    }
    mean_error = create_plots_with_cst_comparison(theta_degrees, F_norm, D_theta, 
                                                 D_theta_dB, cst_theta, cst_D, params)
    print("\n" + "=" * 80)
    print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
    print("=" * 80)
    print("Для очень короткого вибратора (l/λ < 0.1):")
    print("- Диаграмма направленности близка к диполю Герца")
    print("- Максимум излучения при θ = 90°")
    print("- Dmax ≈ 1.5 (1.76 дБ) для идеального диполя")
    if len(cst_theta) > 0:
        print(f"\nСредняя относительная ошибка между теорией и CST: {mean_error:.2f}%")
        if mean_error < 5:
            print("✅ Результаты хорошо согласуются (ошибка < 5%)")
        elif mean_error < 10:
            print("⚠ Результаты удовлетворительно согласуются (ошибка 5-10%)")
        else:
            print("❌ Результаты плохо согласуются (ошибка > 10%)")
        
        print("\nВозможные причины расхождений:")
        print("   - В CST диполь имеет конечный радиус, а в теории - бесконечно тонкий")
        print("   - Погрешности численного расчета в CST")
        print("   - Влияние дискретного порта возбуждения")
        print("   - Конечная точность сетки в CST")
if __name__ == "__main__":
    main()