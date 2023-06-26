import pandas as pd
import numpy as np

def taula_TG(df, var1, var2):
    df = pd.DataFrame({
        'variable1': df[var1],
        'variable2': df[var2],
        'binaria': df['TenYearCHD']
    })

    categorias1 = sorted(df['variable1'].unique())
    categorias2 = sorted(df['variable2'].unique())

    matriz_pagos = np.zeros((len(categorias1), len(categorias2)))


    for i, cat1 in enumerate(categorias1):
        for j, cat2 in enumerate(categorias2):
            sub_df = df[(df['variable1'] == cat1) & (df['variable2'] == cat2)]
            if len(sub_df) > 0:
                matriz_pagos[i, j] = sub_df['binaria'].mean()

    df_pagos = pd.DataFrame(matriz_pagos, index=categorias1, columns=categorias2)

    return df_pagos, categorias1, categorias2






