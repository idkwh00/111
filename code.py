import os, sys, json, math, csv
import numpy as nump
import matplotlib.pyplot as plot
from scipy.special import spherical_jn as s_jn, spherical_yn as s_yn
import xml.etree.ElementTree as ET
from xml.dom import minidom
SPEED_OF_LIGHT = 299792458.0
class VariantParser:
    @staticmethod
    def parse_numeric_value(input_value):
        try:
            cleaned_value = str(input_value).strip()
            return float(cleaned_value)
        except:
            raise TypeError(f"Невозможно преобразовать '{input_value}' в число")
    @classmethod
    def load_csv_variants(cls, file_path):
        variant_collection = {}
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',', skipinitialspace=True)
            print("Заголовки CSV файла:", csv_reader.fieldnames) 
            for row_index, row in enumerate(csv_reader, 1):
                try:
                    variant_id = int(row.get('Вариант', row.get('id', row_index)))
                    diameter = row.get('D', row.get('Диаметр', row.get('diameter', 0)))
                    fmin = row.get('fmin', row.get('fmin, Гц', row.get('frequency_min', 0)))
                    fmax = row.get('fmax', row.get('fmax, Гц', row.get('frequency_max', 0)))
                    if diameter == 0:
                        for key in row.keys():
                            if 'd' in key.lower() or 'диа' in key.lower():
                                diameter = row[key]
                                break
                    if fmin == 0:
                        for key in row.keys():
                            if 'min' in key.lower() or 'мин' in key.lower():
                                fmin = row[key]
                                break
                    if fmax == 0:
                        for key in row.keys():
                            if 'max' in key.lower() or 'макс' in key.lower():
                                fmax = row[key]
                                break
                    print(f"Строка {row_index}: id={variant_id}, fmin='{fmin}', fmax='{fmax}', D='{diameter}'")
                    variant_collection[variant_id] = {
                        'diameter': cls.parse_numeric_value(diameter),
                        'frequency_minimum': cls.parse_numeric_value(fmin),
                        'frequency_maximum': cls.parse_numeric_value(fmax)
                    }
                    print(f"  Преобразовано: D={variant_collection[variant_id]['diameter']}, "
                          f"fmin={variant_collection[variant_id]['frequency_minimum']}, "
                          f"fmax={variant_collection[variant_id]['frequency_maximum']}")
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Ошибка в строке {row_index}: {e}")
                    print(f"Содержимое строки: {row}")
                    continue
        return variant_collection
class RadarCrossSectionSolver:
    def __init__(self, sphere_diameter):
        self.object_diameter = float(sphere_diameter)
        self.object_radius = sphere_diameter * 0.5
    def _determine_series_length(self, wave_parameter):
        if wave_parameter <= 0:
            return 20
        base_component = wave_parameter
        adjustment_component = 4.0 * (wave_parameter ** (1.0 / 3.0))
        minimum_terms = 20
        estimated_terms = int(math.ceil(base_component + adjustment_component + 2.0))
        return max(estimated_terms, minimum_terms)
    def compute_scattering_cross_section(self, operating_frequency):
        if operating_frequency <= 0:
            return 0.0
        signal_wavelength = SPEED_OF_LIGHT / operating_frequency
        wave_number = (2.0 * math.pi) / signal_wavelength
        dimensionless_parameter = wave_number * self.object_radius
        series_length = self._determine_series_length(dimensionless_parameter)
        series_sum = 0 + 0j
        for term_index in range(1, series_length + 1):
            bessel_j = s_jn(term_index, dimensionless_parameter)
            bessel_y = s_yn(term_index, dimensionless_parameter)
            hankel_function = complex(bessel_j, bessel_y)
            bessel_j_prev = s_jn(term_index - 1, dimensionless_parameter)
            bessel_y_prev = s_yn(term_index - 1, dimensionless_parameter)
            hankel_function_prev = complex(bessel_j_prev, bessel_y_prev)
            scattering_coefficient_a = (bessel_j / hankel_function) if hankel_function != 0 else 0
            numerator_component = dimensionless_parameter * bessel_j_prev - term_index * bessel_j
            denominator_component = dimensionless_parameter * hankel_function_prev - term_index * hankel_function
            scattering_coefficient_b = (
                        numerator_component / denominator_component) if denominator_component != 0 else 0
            alternating_sign = 1 if (term_index % 2 == 0) else -1
            weighting_factor = term_index + 0.5
            series_sum += alternating_sign * weighting_factor * (scattering_coefficient_b - scattering_coefficient_a)
        scattering_area = (signal_wavelength ** 2 / math.pi) * (abs(series_sum) ** 2)
        return float(scattering_area)
    def compute_for_frequency_set(self, frequency_array):
        frequency_values = nump.asarray(frequency_array, dtype=float)
        wavelength_values = SPEED_OF_LIGHT / frequency_values
        scattering_results = []
        for single_frequency in frequency_values:
            scattering_results.append(self.compute_scattering_cross_section(float(single_frequency)))
        return wavelength_values.tolist(), scattering_results
class ResultSaver:
    @classmethod
    def export_to_xml(cls, frequencies, wavelengths, scattering_data, variant_id, sphere_diameter, output_file):
        root = ET.Element('RadarCrossSectionResults')
        metadata = ET.SubElement(root, 'Metadata')
        ET.SubElement(metadata, 'VariantID').text = str(variant_id)
        ET.SubElement(metadata, 'SphereDiameter').text = f"{sphere_diameter:.6f}"
        ET.SubElement(metadata, 'NumberOfPoints').text = str(len(frequencies))
        ET.SubElement(metadata, 'SpeedOfLight').text = f"{SPEED_OF_LIGHT:.2f}"
        data_element = ET.SubElement(root, 'DataPoints')
        for i in range(len(frequencies)):
            point = ET.SubElement(data_element, 'Point')
            point.set('index', str(i+1))
            ET.SubElement(point, 'Frequency_Hz').text = f"{frequencies[i]:.6e}"
            ET.SubElement(point, 'Wavelength_m').text = f"{wavelengths[i]:.6e}"
            ET.SubElement(point, 'RCS_m2').text = f"{scattering_data[i]:.6e}"
        xml_string = ET.tostring(root, encoding='utf-8')
        parsed_xml = minidom.parseString(xml_string)
        pretty_xml = parsed_xml.toprettyxml(indent="  ")
        with open(output_file, 'w', encoding='utf-8') as xml_file:
            xml_file.write(pretty_xml)
        return output_file
    @staticmethod
    def create_visualization(frequency_points, scattering_points, variant_id, sphere_diameter):
        plot.figure(figsize=(10, 6))
        plot.semilogy(nump.array(frequency_points) / 1e9, scattering_points,
                      linewidth=2, color='navy')
        plot.title(
            f'Зависимость эффективной площади рассеяния от частоты\nВариант {variant_id}, диаметр = {sphere_diameter} м',
            fontsize=14)
        plot.xlabel('Частота, ГГц', fontsize=12)
        plot.ylabel('ЭПР, м²', fontsize=12)
        plot.grid(True, alpha=0.3, linestyle='--')
        plot.tight_layout()
        plot_filename = f'rcs_variant_{variant_id}.png'
        plot.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plot.show()
        return plot_filename
def run_variant_calculation(variant_id, calculation_parameters):
    sphere_diameter = calculation_parameters['diameter']
    frequency_start = calculation_parameters['frequency_minimum']
    frequency_end = calculation_parameters['frequency_maximum']
    print(f"\n{'=' * 65}")
    print(f"РАСЧЕТ ЭФФЕКТИВНОЙ ПЛОЩАДИ РАССЕЯНИЯ")
    print(f"Вариант: {variant_id}")
    print(f"Диаметр сферы: {sphere_diameter} м")
    print(f"Диапазон частот: {frequency_start:.3e} - {frequency_end:.3e} Гц")
    print('=' * 65)
    frequency_grid = nump.linspace(frequency_start, frequency_end, 400)
    calculator_instance = RadarCrossSectionSolver(sphere_diameter)
    wavelength_results, scattering_results = calculator_instance.compute_for_frequency_set(
        frequency_grid)
    xml_output_filename = f"rcs_variant_{variant_id}.xml"
    ResultSaver.export_to_xml(frequency_grid, wavelength_results, scattering_results,
                              variant_id, sphere_diameter, xml_output_filename)
    print("\nПРЕВЬЮ РЕЗУЛЬТАТОВ (первые 10 точек):")
    print("-" * 85)
    print("№      Частота (Гц)        Длина волны (м)        ЭПР (м²)")
    print("-" * 85)
    for point_index in range(min(10, len(frequency_grid))):
        print(f"{point_index + 1:2d}  {frequency_grid[point_index]:12.6e}  "
              f"{wavelength_results[point_index]:12.6e}  {scattering_results[point_index]:12.6e}")
    ResultSaver.create_visualization(frequency_grid, scattering_results,
                                     variant_id, sphere_diameter)
    print(f"\n✓ Результаты сохранены в файл: {xml_output_filename}")
    print("✓ График сохранен в текущей директории")
    return {
        'frequency_values': frequency_grid,
        'wavelength_values': wavelength_results,
        'scattering_values': scattering_results
    }
def main():
    variant_file_path = "task_rcs_01.csv"
    if not os.path.isfile(variant_file_path):
        working_directory = os.getcwd()
        print(f"ОШИБКА: Файл вариантов '{variant_file_path}' не найден.")
        print(f"Текущая рабочая директория: {working_directory}")
        sys.exit(1)
    variant_set = VariantParser.load_csv_variants(variant_file_path)
    if not variant_set:
        print("ОШИБКА: Не удалось загрузить варианты из CSV файла.")
        print("Убедитесь, что файл содержит правильные заголовки.")
        sys.exit(1)
    print("\nДОСТУПНЫЕ ВАРИАНТЫ ДЛЯ РАСЧЕТА:")
    print("ID  Диаметр (м)     Min частота (Гц)    Max частота (Гц)")
    print("-" * 65)
    for variant_id in sorted(variant_set.keys()):
        variant_data = variant_set[variant_id]
        print(f"{variant_id:2d}  {variant_data['diameter']:12.6g}  "
              f"{variant_data['frequency_minimum']:15.6g}  {variant_data['frequency_maximum']:15.6g}")
    available_ids = sorted(variant_set.keys())
    while True:
        try:
            user_selection = input(f"\nВведите номер варианта для расчета: ").strip()
            if not user_selection:
                print("Ошибка: необходимо указать номер варианта.")
                continue
            selected_variant = int(user_selection)
            if selected_variant in variant_set:
                break
            else:
                print(f"Вариант {selected_variant} не найден.")
        except ValueError:
            print("Ошибка: необходимо ввести целое число.")
    run_variant_calculation(selected_variant, variant_set[selected_variant])
if __name__ == '__main__':
    main()