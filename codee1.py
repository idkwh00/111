import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
class EPRCalculator:
    def __init__(self, D: float, c=3e8, N : int = 200):
        self.c = c
        self.N = N
        self.r = D / 2
    def hankel_three_kind(self, n, k, r):
        return (spherical_jn(n, k * r) + 1j * spherical_yn(n, k * r))
    def an(self, n, k, r):
        return (spherical_jn(n, k * r) / self.hankel_three_kind(n, k, r))
    def bn(self, n, k, r):
        return ((k * r * spherical_jn(n - 1, k * r) - n * spherical_jn(n, k * r)) / (k * r * self.hankel_three_kind(n - 1, k, r) - n * self.hankel_three_kind(n, k, r)))
def sigma(self, lmbd, k, r):
    s = 0
    for n in range(1, self.N + 1):
        s += ((-1)**n * (n + 0.5) * (self.bn(n, k, r) - self.an(n, k, r)))
    return (lmbd**2 / np.pi) * abs(s)**2
def calculate(self, frequencies) -> list:
    sigmas = []
    for f in frequencies:
        lmbd = self.c / f
        k = 2 * np.pi / lmbd
        sigmas.append(self.sigma(lmbd, k, self.r))
    return sigmas
class EPRPlotter:
    @staticmethod
    def save_to_xml(frequencies, sigmas, D, filename="output.xml"):
        root = ET.Element("data")
        for f, s in zip(frequencies, sigmas):
            entry = ET.SubElement(root, "row")
            freq_elem = ET.SubElement(entry, "freq")
            freq_elem.text = str(f)
            lambda_elem = ET.SubElement(entry,"lambda")
            lambda_elem.text = str(3e8 / f)
            sigma_elem = ET.SubElement(entry, "rcs")
            sigma_elem.text = str(s)
        rough_string = ET.tostring(root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent=" ")
        with open('output.xml', "w") as f:
            f.write(pretty_xml)
if __name__=='__main__':
    input_csv = 'task_rcs_01.csv'
    output_xml = 'output.xml'
    variant_numb = 1
    fmin = fmax = D = None
    with open(input_csv, encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            try:
                variant = int(row[0].strip())
                if variant == variant_numb:
                    fmin = float(row[1])
                    fmax = float(row[2])
                    D = float(row[3])
                    break
            except Exception:
                continue
    if fmin is None or fmax is None or D is None:

        raise ValueError(f"Вариант {variant_numb} не найден или данные некорректны.")
frequencies = np.linspace(fmin, fmax, 200)
calculator = EPRCalculator(D=D, N=50)
sigmas = calculator.calculate(frequencies)
EPRPlotter.save_to_xml(frequencies, sigmas,
calculator.r * 2, filename=output_xml)
plt.plot(frequencies, sigmas)
plt.xlabel('Freq')
plt.ylabel('Sigma')
plt.title('sigmas = f(Freq)')
plt.grid(True)
plt.savefig('task1var1.png')
plt.show()