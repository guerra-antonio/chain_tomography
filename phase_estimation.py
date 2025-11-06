# --------------------------------
# Importamos librerias
# --------------------------------
import numpy as np
import torch
import pandas as pd
import re

from tomography import fidelity
from tqdm import tqdm
torch.set_default_dtype(torch.float32)

# --------------------------------
# Definimos nuestro modelo de ML
# --------------------------------
class DiagonalPhase(torch.nn.Module):
    def __init__(self, init_alpha=(0.0, 0.0, 0.0)):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))  

    def phases_matrix_diagonal(self):
        # alpha: (3,) -> φ=[a, b, c, 0]
        phi = torch.cat([self.alpha, torch.zeros(1, dtype=self.alpha.dtype)])  
        v = torch.exp(1j * phi)                                               
        D = torch.diag(v)                            
        return D

    def show(self):
        return self.alpha

    def forward(self, rho_prev):
        D = self.phases_matrix_diagonal()
        # ρ' = D @ ρ @ D†
        rho_next = D @ rho_prev @ D.conj().T

        return rho_next
    
# --------------------------------
# Leemos la data generada por nuestro full_tomography.py
# --------------------------------
def read_complex_matrix(txt: str, n=4, dtype=np.complex64):
    # Eliminar saltos y corchetes
    txt = txt.replace('\n', ' ').replace('[', ' ').replace(']', ' ')
    txt = ' '.join(txt.split())

    # Unir la parte real e imaginaria separadas por espacio
    # Casos cubiertos:
    # "0.23 +0.1j", "-0.12 -0.05j", " 0.24 +0.j", etc.
    txt = re.sub(
        r'(\d)\s*([+\-])\s*(\d)',
        r'\1\2\3',
        txt
    )

    # Parseo robusto
    vals = np.fromstring(txt, sep=' ', dtype=np.complex128)
    if vals.size != n * n:
        raise ValueError(f"Esperaba {n*n} números complejos, obtuve {vals.size}.")
    return vals.reshape(n, n).astype(dtype)

# --------------------------------
# Hacemos la estimación
# --------------------------------
df_main = pd.read_excel("tomography_results.xlsx")
df_tomography = df_main["rho"]
n_experiments = df_tomography.shape[0] // 50
n_epochs = 2000

# Preparamos nuestro dataframe para guardar los datos
df_phases = pd.DataFrame(columns=["loss", "fases", "fid_primervecino", "fid_todosvecino", "file"], dtype=object)

for exp in tqdm(range(1, n_experiments), desc="Estimando fases para el experimento"):
    # Leemos las tomografías de un experimento particular
    df_exp = df_tomography[50 * (exp - 1): 50 * (exp)]
    
    # Guardamos esos rho's en un arreglo de torch para usar nuestro modelo de ML
    datos = []
    n_datos = df_exp.shape[0]
    for step in range(n_datos):
        rho_str = df_exp.iloc[step]
        rho = read_complex_matrix(txt = rho_str)
        datos.append(rho)

    datos = torch.tensor(np.array(datos), dtype=torch.complex64)

    # Ejecutamos la estimación
    model = DiagonalPhase()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, requires_grad=True)

        # acumulamos todas las pérdidas dentro del grafo
        for i in range(1, n_datos):
            rho_step = datos[i - 1, :, :]
            rho_next = datos[i, :, :]
            rho_calc = model(rho_step)

            total_loss = total_loss + torch.sum(torch.abs(rho_calc - rho_next))

        # promedio y backward
        loss_mean = total_loss / (n_datos - 1)
        loss_mean.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss_mean.item())
    losses = np.array(losses)

    # Con las fases estimadas calculamos la fideldad a primeros vecinos
    fids_pv = []

    for data_ in range(1, n_datos):
        rho_prev = datos[data_ - 1, :, :]                 # (4,4)
        rho_next = datos[data_, :, :]                   # (4,4)
        rho_calc = model(rho_prev)         # (4,4)

        # Convertimos a np.array para calcular la fidelidad
        rho_next = rho_next.detach().numpy()
        rho_calc = rho_calc.detach().numpy()
        fid = fidelity(rho=rho_next, sigma=rho_calc)

        fids_pv.append(fid)
    fids_pv = np.array(fids_pv)

    # Con las fases estimadas calculamos la fideldad con todos los vecinos
    fids_fv = []
    rho_initial = datos[0, :, :]
    D = model.phases_matrix_diagonal()

    for data_ in range(1, n_datos):
        rho_step = datos[data_, :, :]                   # (4,4)
        rho_calc = D**data_ @ rho_initial @ (D.conj().T)**data_        # (4,4)
        
        # Convertimos a np.array para calcular la fidelidad
        rho_step = rho_step.detach().numpy()
        rho_calc = rho_calc.detach().numpy()
        fid = fidelity(rho=rho_step, sigma=rho_calc)

        fids_fv.append(fid)
    fids_fv = np.array(fids_fv)

    # Guardamos los datos 
    df_phases.loc[len(df_phases)] = [losses, model.alpha.detach().numpy() ,fids_pv, fids_fv, df_main["file"].iloc[50 * exp]]

# Exportamos la data 
df_phases.to_excel("phase_estimation_results.xlsx")