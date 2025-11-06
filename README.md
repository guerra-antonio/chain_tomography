# 🧮 Quantum State Tomography (4D)

Este repositorio implementa un esquema completo de **tomografía cuántica** para reconstruir matrices densidad de sistemas bidimensionales compuestos (dos qubits, dimensión 4).  
El proceso se basa en medidas en **bases mutuamente no sesgadas (MUBs)** y en optimización utilizando **CVXPY**.

---

## 📁 Estructura del proyecto

```
├── tomography.py              # Rutinas de tomografía y fidelidad cuántica
├── full_tomography.py         # Script principal que procesa todos los experimentos
├── tomography_results.xlsx    # Resultados exportados (pureza, fidelidad, rho)
└── data_exp/                  # Carpeta esperada con los datos experimentales (.txt o .csv)
```

---

## ⚙️ Dependencias

El proyecto fue desarrollado y probado con las siguientes versiones:

```
python == 3.9.19
numpy == 1.24.4
pandas == 1.7.3
cvxpy == 1.7.3
scipy == 1.13.1
tqdm == 4.67.1
```

---

## 🧠 Descripción de los módulos

### `tomography.py`

Contiene las rutinas principales:

- **Bases mutuamente no sesgadas (MUB1–MUB5)** definidas para dimensión 4.  
- `extract_data(df, n)`: obtiene los proyectores y probabilidades de cada bloque experimental.  
- `tomography(data)`: resuelve el problema de optimización por mínimos cuadrados sujeto a las restricciones de traza unitaria y semipositividad.
- `fidelity(rho, sigma)`: calcula la fidelidad cuántica entre dos matrices densidad reconstruidas.

---

### `full_tomography.py`

Automatiza la reconstrucción completa de los estados:

1. Lee todos los archivos en la carpeta `data_exp/`.  
2. Filtra las columnas relevantes de cada dataset.  
3. Ejecuta `tomography()` sobre cada conjunto.  
4. Calcula la **pureza** y la **fidelidad** respecto al primer estado reconstruido.  
5. Guarda los resultados consolidados en `tomography_results.xlsx`.

---

## 🧪 Ejecución

1. Coloca tus archivos experimentales (`.txt` o `.csv`) dentro de la carpeta `data_exp/`.  
2. Ejecuta el script principal:

   ```bash
   python full_tomography.py
   ```

3. Se generará automáticamente el archivo `tomography_results.xlsx` con las siguientes columnas:

| Columna     | Descripción |
|--------------|-------------|
| `pureza`     | Tr(ρ²): medida de mezcla del estado reconstruido |
| `fidelidad`  | Fidelidad respecto al estado de referencia (inicial) |
| `ρ`          | Matriz densidad reconstruida |
| `file`       | Nombre del archivo de datos |

---

## 🧩 Referencias

- M. A. Nielsen & I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press (2000).  
- Teoría de **Bases Mutuamente No Sesgadas (MUBs)** en espacios de Hilbert finitos.  
- [Documentación de CVXPY](https://www.cvxpy.org/)
