# Importamos librerias necesarias
import os

import pandas as pd
import numpy as np

from tomography import tomography, fidelity
from tqdm import tqdm

# ------------------------------------------------------------
# Hacemos la tomografía para todos los datos ubicados en la carpeta data_exp
# ------------------------------------------------------------
main_file   = "data_exp"
files       = os.listdir(main_file)
df_exps     = pd.DataFrame({
    "pureza": pd.Series(dtype=float),
    "fidelidad": pd.Series(dtype=float),
    "rho": pd.Series(dtype=object),
    "file": pd.Series(dtype=str)
    })

for file in tqdm(files, desc="Haciendo las tomografías de los experimentos"):
    df = pd.read_csv(f"data_exp/{file}", sep=r"\s+", header=None)
    df.columns = [f"col{i}" for i in range(1, df.columns.shape[0] + 1)]

    # Mantener 7,8,9,10 y 14 (base-1) ➜ índices 0-based
    cols_keep = [5, 6, 7, 8, 12]
    df = df.iloc[:, cols_keep]
    df = df[: 200]

    # Obtenemos los rho's
    tomography_results = tomography(data = df)
    rho_0 = tomography_results[0]

    # Guardamos los datos
    for _ in range(len(tomography_results)):
        rho     = tomography_results[_]
        purity  = np.trace(rho @ rho).real
        fid     = fidelity(rho = rho_0, sigma = rho)

        # Definimos la fila a agregar
        new_row = {
            "pureza" : purity,
            "fidelidad" : fid,
            "rho" : rho,
            "file" : file
        }

        # Agregamos los datos a nuestro df_main
        df_exps = pd.concat([df_exps, pd.DataFrame([new_row])], ignore_index=True)

# Guardamos nuestra data
df_exps.to_excel("tomography_results.xlsx")