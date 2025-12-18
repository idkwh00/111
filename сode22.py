import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
def calculate_vibrator_pattern(l_over_lambda, theta_degrees):
    """Расчет ДН симметричного вибратора"""
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
    """Загрузка данных из файла CST"""
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
def compare_with_cst(theory_theta, theory_D, cst_theta, cst_D, params):
    """Сравнение теоретических и CST данных"""
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(theory_theta, theory_D, 'b-', linewidth=2.5, 
             label=f'Теория, Dmax = {params["D_max_theory"]:.4f}')
    mask = (cst_theta <= 180)  
    unique_indices = np.arange(0, len(cst_theta[mask]), 5)
    ax1.plot(cst_theta[mask][unique_indices], cst_D[mask][unique_indices], 
             'ro', markersize=6, alpha=0.7, label=f'CST, Dmax = {params["D_max_cst"]:.4f}')
    ax1.set_xlabel('Угол θ, градусы', fontsize=12)
    ax1.set_ylabel('Коэффициент направленного действия, D(θ)', fontsize=12)
    ax1.set_title('Сравнение КНД: Теория vs CST (линейная шкала)', fontsize=13)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, max(max(theory_D), max(cst_D[mask])) * 1.1)
    ax2 = plt.subplot(2, 2, 2)
    theory_D_db = 10 * np.log10(theory_D)
    theory_D_db[~np.isfinite(theory_D_db)] = -100
    ax2.plot(theory_theta, theory_D_db, 'g-', linewidth=2.5, 
             label=f'Теория, Dmax = {params["D_max_dB_theory"]:.2f} дБ')
    cst_D_db = 10 * np.log10(cst_D[mask])
    cst_D_db[~np.isfinite(cst_D_db)] = -100 
    ax2.plot(cst_theta[mask][unique_indices], cst_D_db[unique_indices], 
             'ms', markersize=6, alpha=0.7, label=f'CST, Dmax = {params["D_max_dB_cst"]:.2f} дБ')
    ax2.set_xlabel('Угол θ, градусы', fontsize=12)
    ax2.set_ylabel('Коэффициент направленного действия, D(θ), дБ', fontsize=12)
    ax2.set_title('Сравнение КНД: Теория vs CST (логарифмическая шкала)', fontsize=13)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_xlim(0, 180)
    ax2.set_ylim(-50, 5)
    ax3 = plt.subplot(2, 2, 3)
    cst_D_interp = np.interp(theory_theta, cst_theta[mask], cst_D[mask])
    valid_mask = (theory_D > 1e-6) & (cst_D_interp > 1e-6)  
    rel_error = np.abs(theory_D[valid_mask] - cst_D_interp[valid_mask]) / theory_D[valid_mask] * 100
    ax3.plot(theory_theta[valid_mask], rel_error, 'r-', linewidth=2.5)
    ax3.fill_between(theory_theta[valid_mask], 0, rel_error, alpha=0.3, color='red')
    ax3.set_xlabel('Угол θ, градусы', fontsize=12)
    ax3.set_ylabel('Относительная ошибка, %', fontsize=12)
    ax3.set_title('Относительная ошибка между теорией и CST', fontsize=13)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 180)
    ax3.set_ylim(0, max(rel_error) * 1.1)
    mean_error = np.mean(rel_error)
    ax3.axhline(y=mean_error, color='blue', linestyle='--', 
                label=f'Средняя ошибка: {mean_error:.2f}%')
    ax3.legend(loc='upper right', fontsize=11)
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    F_norm = theory_D / np.max(theory_D) 
    theta_rad = np.radians(theory_theta)
    ax4.plot(theta_rad, F_norm, 'b-', linewidth=2.5, alpha=0.7, label='Теория')
    cst_F_norm = cst_D[mask] / np.max(cst_D[mask])
    ax4.plot(np.radians(cst_theta[mask]), cst_F_norm, 'r--', linewidth=1.5, alpha=0.7, label='CST')
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_rmax(1.1)
    ax4.grid(True, alpha=0.5, linestyle='--')
    ax4.set_title('Диаграмма направленности (полярные координаты)', pad=20, fontsize=12)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.suptitle(f'Сравнение теоретических расчетов с CST Studio Suite\n'
                 f'Симметричный вибратор, 2l/λ = {params["2l_over_lambda"]:.3f}, f = {params["f_ghz"]:.1f} ГГц', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show() 
    return rel_error, mean_error
def main():
    f = 1.0e9  
    c = 3.0e8  
    l_over_lambda = 0.01 / 2  
    print("=" * 80)
    print("СРАВНЕНИЕ ТЕОРЕТИЧЕСКИХ РАСЧЕТОВ С CST STUDIO SUITE")
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
    cst_filename = "dipol132.txt" 
    cst_theta, cst_D = load_cst_data(cst_filename)
    mask = (cst_theta <= 180)
    D_max_cst = np.max(cst_D[mask])
    D_max_dB_cst = 10 * np.log10(D_max_cst)
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    print(f"{'Параметр':<25} | {'Теория':<15} | {'CST':<15} | {'Разница':<15}")
    print("-" * 80)
    print(f"{'Максимальный КНД (разы)':<25} | {D_max_theory:<15.6f} | {D_max_cst:<15.6f} | {abs(D_max_theory - D_max_cst):<15.6f}")
    print(f"{'Максимальный КНД (дБ)':<25} | {D_max_dB_theory:<15.6f} | {D_max_dB_cst:<15.6f} | {abs(D_max_dB_theory - D_max_dB_cst):<15.6f}")
    D_theoretical = 1.5
    D_theoretical_dB = 10 * np.log10(D_theoretical)
    print(f"\n{'Теорет. для короткого диполя':<25} | {D_theoretical:<15.6f} | -{'':<14} | -{'':<14}")
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
    rel_error, mean_error = compare_with_cst(theta_degrees, D_theta, cst_theta, cst_D, params)
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ В КЛЮЧЕВЫХ ТОЧКАХ:")
    print("=" * 80)
    print(f"{'θ, °':^8} | {'Теория D':^12} | {'CST D':^12} | {'Ошибка, %':^12} | {'Разница':^12}")
    print("-" * 80)
    key_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180]
    for angle in key_angles:
        idx_theory = np.argmin(np.abs(theta_degrees - angle))
        D_theory_val = D_theta[idx_theory] if idx_theory < len(D_theta) else 0
        if angle <= 180:
            idx_near = np.where(cst_theta <= 180)[0]
            if len(idx_near) > 0:
                D_cst_val = np.interp(angle, cst_theta[idx_near], cst_D[idx_near])
                if D_theory_val > 1e-6 and D_cst_val > 1e-6:
                    error = abs(D_theory_val - D_cst_val) / D_theory_val * 100
                    diff = abs(D_theory_val - D_cst_val)
                else:
                    error = 0
                    diff = 0
                
                print(f"{angle:^8} | {D_theory_val:^12.6f} | {D_cst_val:^12.6f} | {error:^12.2f} | {diff:^12.6f}")
    print("\n" + "=" * 80)
    print("ВЫВОДЫ:")
    print("=" * 80)
    print(f"1. Максимальный КНД из CST: {D_max_cst:.6f} ({D_max_dB_cst:.6f} дБ)")
    print(f"2. Теоретический максимум: {D_max_theory:.6f} ({D_max_dB_theory:.6f} дБ)")
    print(f"3. Теоретическое значение для идеального диполя: 1.500000 (1.760913 дБ)")
    print(f"4. Средняя относительная ошибка: {mean_error:.2f}%")
    if mean_error < 5:
        print(f"5. ✅ Результаты хорошо согласуются (ошибка < 5%)")
    elif mean_error < 10:
        print(f"5. ⚠ Результаты удовлетворительно согласуются (ошибка 5-10%)")
    else:
        print(f"5. ❌ Результаты плохо согласуются (ошибка > 10%)")
    print("\n6. Возможные причины расхождений:")
    print("   - В CST диполь имеет конечный радиус, а в теории - бесконечно тонкий")
    print("   - Погрешности численного расчета в CST")
    print("   - Влияние дискретного порта возбуждения")
    print("   - Конечная точность сетки в CST")
if __name__ == "__main__":
    main()