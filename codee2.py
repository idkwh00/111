import numpy as np
import matplotlib.pyplot as plt
def E(theta):
    num = np.cos(k * l * np.cos(theta)) - np.cos(k * l)
    den = np.sin(theta)
    return num/den
def F(theta):
    return abs(E(theta)) / abs(E(theta).max())
def Dmax(theta):
    formula = (F(theta)**2 * np.sin(theta))
    return 2 / np.trapezoid(formula, theta)
def D(theta):
    return F(theta)**2 * Dmax(theta)
def creating_plot(d_times, d_dB, theta):
    fig, axs = plt.subplots(2, 2, figsize=(12,10), subplot_kw={'polar': False})
    fig.suptitle('D(Theta)')
    axs[0,0].plot(theta, d_times, color='blue')
    axs[0,0].set_title("КНД (разы, декарт)")
    axs[0,0].set_xlabel("θ (рад)")
    axs[0,0].set_ylabel("D(θ)")
    axs[0,0].grid(True)
    axs[0,1].plot(theta, d_dB, color='red')
    axs[0,1].set_title("КНД (дБ, декарт)")
    axs[0,1].set_xlabel("θ (рад)")
    axs[0,1].set_ylabel("D(θ) [дБ]")
    axs[0,1].grid(True)
    axs[1,0] = plt.subplot(2,2,3, polar=True)
    axs[1,0].plot(theta, d_times, color='blue')
    axs[1,0].set_title("КНД (разы, поляр)")
    axs[1,1] = plt.subplot(2,2,4, polar=True)
    axs[1,1].plot(theta, d_dB, color='red')
    axs[1,1].set_title("КНД (дБ, поляр)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.savefig('task2var1.png')
def main():
    global l, k
    f = 1 * 10 ** 9
    lmbd = 3 * 10 ** 8 / f
    l = 0.01 * lmbd / 2
    k = 2 * np.pi / lmbd

    theta = np.linspace(1e-9, np.pi-(1e-9), 2000)
    print(f'{Dmax(theta=theta):.3f} times\n{10 * np.log10(Dmax(theta=theta)):.3f} dB')
    creating_plot(d_times=D(theta), d_dB=10*np.log10(D(theta) + 1e-9), theta=theta)
    d_times = D(theta)
    d_db = 10*np.log10(D(theta) + 1e-9)
    with open('analyse_results.txt', 'w', encoding='utf-8') as file:
        file.write('theta d_times d_db\n')
        for i in range(len(theta)):
            file.write(f'{theta[i]} {d_times[i]} {d_db[i]}\n')
if __name__=="__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
def results_from_py():
    with open('analyse_results.txt', 'r', encoding='utf-8') as file:
        file.readline()
        axis =[[],[],[]]
        for line in file:
            theta, d_times, d_db = map(float, file.readline().split())
            axis[0].append(theta)
            axis[1].append(d_db)
            axis[2].append(d_times)
        return axis
def results_from_CST():
    with open('resFF.txt', 'r', encoding='utf-8') as file:
        file.readline()
        file.readline()
        axis = [[],[],[]]
        for line in file:
            Theta, Phi, Direction, d_db, Phase, Phi, Phase_Phi, Ratio = map(float, file.readline().split())
            axis[0].append(np.deg2rad(Theta))
            axis[1].append(d_db)
            axis[2].append(10**(d_db/10))
        return axis
def creating_plot(cst, python):
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Difference between\nPython and CST')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax1.plot(cst[0], cst[1], label='CST')
    ax1.plot(python[0], python[1], label='Python')
    ax1.set_title('D, dBi')
    ax1.set_xlabel("Theta")
    ax1.set_ylabel("dBi")
    ax1.grid(True)
    ax1.legend()
    ax2.plot(cst[0], cst[2], label='CST')
    ax2.plot(python[0], python[2], label='Python')
    ax2.set_title('D, times')
    ax2.set_xlabel("Theta")
    ax2.set_ylabel("times")
    ax2.grid(True)
    ax2.legend()
    ax3 = fig.add_subplot(2, 2, 3, polar=True)
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    ax3.plot(cst[0], cst[1], label='CST')
    ax3.plot(python[0], python[1], label='Python')
    ax3.set_title('D, dBi (polar)')
    ax3.grid(True)
    ax3.legend()
    ax4.plot(cst[0], cst[2], label='CST')
    ax4.plot(python[0], python[2], label='Python')
    ax4.set_title('D, times (polar)')
    ax4.grid(True)
    ax4.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def main():
    creating_plot(cst=results_from_CST(), python=results_from_py())
if __name__=='__main__':
    main()