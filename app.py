import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import combinations

st.set_page_config(
    page_title="Resolutor de Programación Lineal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos de la interfaz
ACENTO = "#0F766E"  # color principal
OPTIMO = "#B45309"  # solución óptima
PALETA_RESTRICCIONES = ["#0F766E", "#6D28D9", "#0369A1", "#BE123C"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown, .stButton button, label {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}

    .block-container {{
        padding-top: 2rem;
        max-width: 1200px;
    }}

    /* Encabezado */
    .encabezado {{
        border-left: 5px solid {ACENTO};
        padding: 0.2rem 0 0.2rem 1rem;
        margin-bottom: 1.5rem;
    }}
    .encabezado h1 {{
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
        padding: 0;
    }}
    .encabezado p {{
        margin: 0.25rem 0 0 0;
        opacity: 0.75;
    }}

    /* Barra lateral */
    section[data-testid="stSidebar"] h2 {{
        font-size: 1.25rem;
        font-weight: 600;
    }}

    /* Botón principal */
    .stButton button[kind="primary"] {{
        background-color: {ACENTO};
        border: none;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border-radius: 8px;
    }}
    .stButton button[kind="primary"]:hover {{
        background-color: #115E59;
    }}

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 500;
        padding: 0.5rem 1rem;
    }}

    /* Métricas */
    div[data-testid="stMetric"] {{
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }}
    div[data-testid="stMetricValue"] {{
        font-weight: 600;
    }}

    /* Expansores de iteraciones */
    div[data-testid="stExpander"] {{
        border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Valor de M para las variables artificiales
M = 100000.0
TOL = 1e-6


def resaltar_fila_z(fila):
    # Resalta la fila Z
    if fila.name == 0:
        return ["font-weight: 600; background-color: rgba(15, 118, 110, 0.18)"] * len(fila)
    return [""] * len(fila)


def termino(coef, var, primero=False):
    mag = abs(coef)
    cuerpo = var if abs(mag - 1) < 1e-12 else f"{mag:g}{var}"
    if primero:
        return ("-" if coef < 0 else "") + cuerpo
    return f" {'-' if coef < 0 else '+'} {cuerpo}"


def expresion(pares):
    # Expresión lineal en LaTeX a partir de pares (coeficiente, variable)
    texto = ""
    for coef, var in pares:
        if abs(coef) < 1e-12:
            continue
        texto += termino(coef, var, primero=(texto == ""))
    return texto if texto else "0"


def mostrar_modelo(opt, c, A, signos, b):
    # Formatear la funcion objetivo
    tipo = "Maximizar" if opt == "Max" else "Minimizar"
    st.latex(rf"\text{{{tipo}}} \quad Z = {c[0]}x_1 + {c[1]}x_2")

    st.markdown("**Sujeto a:**")
    for i in range(3):
        signo_tex = r"\le" if signos[i] == "<=" else r"\ge" if signos[i] == ">=" else "="
        st.latex(rf"{A[i][0]}x_1 + {A[i][1]}x_2 {signo_tex} {b[i]}")
    st.latex(r"x_1, x_2 \ge 0")


# Método gráfico

def calcular_intersecciones(A, b):
    # Definir los coeficientes de las 5 rectas (3 restricciones + 2 ejes)
    lineas_A = list(A) + [[1, 0], [0, 1]]
    lineas_b = list(b) + [0, 0]

    puntos = []
    # Se evita redondear aquí: el redondeo descartaba vértices factibles
    for (i, j) in combinations(range(5), 2):
        matriz = np.array([lineas_A[i], lineas_A[j]], dtype=float)
        vector = np.array([lineas_b[i], lineas_b[j]], dtype=float)
        try:
            punto = np.linalg.solve(matriz, vector)
            puntos.append(punto)
        except np.linalg.LinAlgError:
            pass # Lineas paralelas
    return puntos


def es_factible(punto, A, signos, b):
    x1, x2 = punto
    if x1 < -TOL or x2 < -TOL:
        return False
    for i in range(3):
        valor = A[i][0] * x1 + A[i][1] * x2
        if signos[i] == "<=" and valor > b[i] + TOL: return False
        if signos[i] == ">=" and valor < b[i] - TOL: return False
        if signos[i] == "=" and abs(valor - b[i]) > TOL: return False
    return True


def direcciones_recesion(A, signos):
    # Rayos extremos de la región factible (direcciones d >= 0 con A d compatible con los signos)
    candidatas = [(1.0, 0.0), (0.0, 1.0)]
    for fila in A:
        norma = np.hypot(fila[0], fila[1])
        if norma > 0:
            d = (-fila[1] / norma, fila[0] / norma)
            candidatas += [d, (-d[0], -d[1])]
    validas = []
    for d in candidatas:
        if d[0] < -1e-9 or d[1] < -1e-9:
            continue
        ok = True
        for i in range(3):
            v = A[i][0] * d[0] + A[i][1] * d[1]
            if signos[i] == "<=" and v > 1e-9: ok = False
            elif signos[i] == ">=" and v < -1e-9: ok = False
            elif signos[i] == "=" and abs(v) > 1e-9: ok = False
        if ok:
            validas.append(d)
    return validas


def objetivo_no_acotado(opt, c, rayos):
    # La función objetivo mejora indefinidamente si mejora a lo largo de algún rayo
    for d in rayos:
        mejora = c[0] * d[0] + c[1] * d[1]
        if (opt == "Max" and mejora > 1e-9) or (opt == "Min" and mejora < -1e-9):
            return True
    return False


def finalizar_figura(fig, ax, x_lim, y_lim):
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)


def metodo_grafico(opt, c, A, signos, b):
    st.subheader("Método Gráfico")
    intersecciones = calcular_intersecciones(A, b)
    puntos_factibles = [p for p in intersecciones if es_factible(p, A, signos, b)]

    # Remover duplicados
    puntos_unicos = []
    for p in puntos_factibles:
        p = np.where(np.abs(p) < 1e-9, 0.0, p)
        if not any(np.allclose(p, pu, atol=TOL) for pu in puntos_unicos):
            puntos_unicos.append(p)

    if not puntos_unicos:
        st.error("El problema es infactible. No existe región factible.")
        return {"estado": "infactible"}

    fig, ax = plt.subplots(figsize=(8, 6))
    max_x1 = max([p[0] for p in intersecciones if p[0] >= 0], default=10) * 1.2
    max_x2 = max([p[1] for p in intersecciones if p[1] >= 0], default=10) * 1.2
    x_lim = max(max_x1, 10)
    y_lim = max(max_x2, 10)
    x_vals = np.linspace(0, x_lim, 400)

    # Graficar restricciones
    for i in range(3):
        color_r = PALETA_RESTRICCIONES[i]
        if A[i][1] != 0:
            y_vals = (b[i] - A[i][0] * x_vals) / A[i][1]
            ax.plot(x_vals, y_vals, color=color_r, linewidth=2,
                    label=f'R{i+1}: {A[i][0]}x1 + {A[i][1]}x2 = {b[i]}')
        elif A[i][0] != 0:
            ax.axvline(x=b[i]/A[i][0], label=f'R{i+1}: x1 = {b[i]/A[i][0]}',
                       color=color_r, linewidth=2)

    # Rellenar region factible
    rayos = direcciones_recesion(A, signos)
    puntos_arr = np.array(puntos_unicos)
    if rayos:
        # Región no acotada: se rellena evaluando las restricciones en una malla
        xs = np.linspace(0, x_lim, 300)
        ys = np.linspace(0, y_lim, 300)
        X, Y = np.meshgrid(xs, ys)
        mascara = np.ones_like(X, dtype=bool)
        paso = max(x_lim, y_lim) / 299
        for i in range(3):
            v = A[i][0] * X + A[i][1] * Y
            if signos[i] == "<=": mascara &= v <= b[i] + 1e-9
            elif signos[i] == ">=": mascara &= v >= b[i] - 1e-9
            else: mascara &= np.abs(v - b[i]) <= 0.5 * paso * np.hypot(A[i][0], A[i][1])
        if mascara.any():
            ax.contourf(X, Y, mascara.astype(float), levels=[0.5, 1.5], colors=[ACENTO], alpha=0.25)
        ax.fill([], [], color=ACENTO, alpha=0.25, label='Región Factible (no acotada)')
        ax.plot(puntos_arr[:,0], puntos_arr[:,1], 'o', color='#1F2937', markersize=8, label='Vértices factibles')
    elif len(puntos_unicos) >= 3:
        try:
            hull = ConvexHull(puntos_arr)
            for simplex in hull.simplices:
                ax.plot(puntos_arr[simplex, 0], puntos_arr[simplex, 1], '-', color='#1F2937', linewidth=1.5)
            ax.fill(puntos_arr[hull.vertices,0], puntos_arr[hull.vertices,1], alpha=0.25,
                    color=ACENTO, label='Región Factible')
        except Exception:
            # Región degenerada (puntos alineados)
            ax.plot(puntos_arr[:,0], puntos_arr[:,1], 'o-', color='#1F2937', markersize=8, label='Puntos Factibles')
    else:
        ax.plot(puntos_arr[:,0], puntos_arr[:,1], 'o', color='#1F2937', markersize=8, label='Puntos Factibles')

    # Función objetivo no acotada: no hay óptimo finito
    if rayos and objetivo_no_acotado(opt, c, rayos):
        finalizar_figura(fig, ax, x_lim, y_lim)
        st.error("El problema es no acotado: la función objetivo mejora indefinidamente dentro de la región factible.")
        return {"estado": "no acotado"}

    # Evaluar vertices
    mejor_z = -np.inf if opt == "Max" else np.inf
    mejor_punto = None
    datos_vertices = []

    for p in puntos_unicos:
        z = c[0]*p[0] + c[1]*p[1]
        datos_vertices.append({"x1": p[0], "x2": p[1], "Z": z})
        if opt == "Max" and z > mejor_z:
            mejor_z = z
            mejor_punto = p
        elif opt == "Min" and z < mejor_z:
            mejor_z = z
            mejor_punto = p

    # Marcar el punto optimo y la recta de la función objetivo que pasa por él
    if mejor_punto is not None:
        if c[1] != 0:
            ax.plot(x_vals, (mejor_z - c[0] * x_vals) / c[1], '--', color=OPTIMO, linewidth=1.5,
                    label=f'Z = {mejor_z:.2f}')
        elif c[0] != 0:
            ax.axvline(x=mejor_z / c[0], linestyle='--', color=OPTIMO, linewidth=1.5,
                       label=f'Z = {mejor_z:.2f}')
        ax.plot(mejor_punto[0], mejor_punto[1], '*', color=OPTIMO, markeredgecolor='white',
                markersize=18, label='Óptimo')

    finalizar_figura(fig, ax, x_lim, y_lim)

    st.markdown("**Evaluación de vértices factibles:**")
    st.dataframe(pd.DataFrame(datos_vertices).round(4), hide_index=True)
    st.success(f"**Solución Óptima Gráfica:** x1 = {mejor_punto[0]:.2f}, x2 = {mejor_punto[1]:.2f}, Z = {mejor_z:.2f}")
    return {"estado": "optimo", "x1": float(mejor_punto[0]), "x2": float(mejor_punto[1]), "Z": float(mejor_z)}


# Método simplex

def preparar_restricciones(A, signos, b):
    # Si b < 0 se multiplica la restricción por -1 e invierte el signo (Lado derecho >= 0)
    A2 = np.array(A, dtype=float).copy()
    b2 = np.array(b, dtype=float).copy()
    s2 = list(signos)
    cambios = []
    for i in range(3):
        if b2[i] < 0:
            A2[i] *= -1
            b2[i] *= -1
            s2[i] = {"<=": ">=", ">=": "<=", "=": "="}[s2[i]]
            cambios.append(i + 1)
    return A2, s2, b2, cambios


def columnas_visibles(signos):
    # Solo las variables que existen en el sistema
    cols = ['Z', 'x1', 'x2']
    cols += [f's{i+1}' for i in range(3) if signos[i] in ("<=", ">=")]
    cols += [f'a{i+1}' for i in range(3) if signos[i] in (">=", "=")]
    return cols + ['Lado derecho']


def mostrar_sistema(opt, c, A, signos, b):
    st.markdown("**Sistema en forma de ecuación** (holgura, superávit y artificiales):")
    hay_art = False
    for i in range(3):
        pares = [(A[i][0], "x_1"), (A[i][1], "x_2")]
        if signos[i] == "<=":
            pares.append((1, f"s_{i+1}"))
        if signos[i] == ">=":
            pares.append((-1, f"s_{i+1}"))
        if signos[i] in (">=", "="):
            pares.append((1, f"a_{i+1}"))
            hay_art = True
        st.latex(rf"{expresion(pares)} = {b[i]:g}")

    # Fila Z: Z - c1 x1 - c2 x2 (+M a para Max, -M a para Min) = 0
    signo_M = -1 if opt == "Min" else 1
    pares_z = [(1, "Z"), (-c[0], "x_1"), (-c[1], "x_2")]
    pares_z += [(signo_M, f"Ma_{i+1}") for i in range(3) if signos[i] in (">=", "=")]
    st.latex(rf"{expresion(pares_z)} = 0")
    if hay_art:
        st.caption(f"En los cálculos numéricos se usa M = {M:g}.")


def inicializar_simplex(opt, c, A, signos, b, ajustar=True):
    columnas = ['Z', 'x1', 'x2', 's1', 's2', 's3', 'a1', 'a2', 'a3', 'Lado derecho']
    tabla = np.zeros((4, 10))
    vars_basicas = [''] * 3

    # Llenar matriz de coeficientes
    for i in range(3):
        tabla[i+1, 1] = A[i][0]
        tabla[i+1, 2] = A[i][1]
        tabla[i+1, 9] = b[i]

        if signos[i] == "<=":
            tabla[i+1, 3+i] = 1
            vars_basicas[i] = f's{i+1}'
        elif signos[i] == ">=":
            tabla[i+1, 3+i] = -1
            tabla[i+1, 6+i] = 1
            vars_basicas[i] = f'a{i+1}'
        elif signos[i] == "=":
            tabla[i+1, 6+i] = 1
            vars_basicas[i] = f'a{i+1}'

    # Fila Z inicial: Z - c1 x1 - c2 x2 (+M a para Max, -M a para Min)
    tabla[0, 0] = 1
    tabla[0, 1], tabla[0, 2] = -c[0], -c[1]
    for i in range(3):
        if vars_basicas[i].startswith('a'):
            tabla[0, 6+i] = M if opt == "Max" else -M

    # Ajustar fila Z para anular los coeficientes de las artificiales básicas
    if ajustar:
        for i in range(3):
            if vars_basicas[i].startswith('a'):
                if opt == "Max":
                    tabla[0, :] -= M * tabla[i+1, :]
                else:
                    tabla[0, :] += M * tabla[i+1, :]

    return tabla, columnas, vars_basicas


def tabla_a_df(tabla, columnas, vars_basicas, visibles):
    df = pd.DataFrame(tabla, columns=columnas)
    df.insert(0, 'Base', ['Z'] + vars_basicas)
    return df[['Base'] + visibles]


def mostrar_tabla(destino, tabla, columnas, vars_basicas, visibles):
    df = tabla_a_df(tabla, columnas, vars_basicas, visibles)
    destino.dataframe(df.style.format(precision=2).apply(resaltar_fila_z, axis=1), hide_index=True)


def interpretar_simplex(opt, signos, cambios, tabla, columnas, vars_basicas, visibles, valor, z_opt, hay_art):
    st.subheader("Interpretación de los resultados")
    objetivo = "máximo" if opt == "Max" else "mínimo"
    lineas = [f"- El {objetivo} de Z es **{z_opt:.2f}**, en el punto (x1, x2) = ({valor('x1'):.2f}, {valor('x2'):.2f})."]

    for i in range(3):
        nota = " (multiplicada por -1)" if (i + 1) in cambios else ""
        if signos[i] == "=":
            lineas.append(f"- Restricción {i+1}{nota}: se cumple con igualdad.")
            continue
        v = valor(f's{i+1}')
        if signos[i] == "<=":
            if abs(v) < 1e-6:
                lineas.append(f"- Restricción {i+1}{nota}: activa. El recurso se agota (holgura = 0).")
            else:
                lineas.append(f"- Restricción {i+1}{nota}: no activa. Sobran {v:.2f} unidades (holgura s{i+1} = {v:.2f}).")
        else:
            if abs(v) < 1e-6:
                lineas.append(f"- Restricción {i+1}{nota}: activa. Se alcanza justo el mínimo requerido (superávit = 0).")
            else:
                lineas.append(f"- Restricción {i+1}{nota}: no activa. Se excede el mínimo en {v:.2f} unidades (superávit s{i+1} = {v:.2f}).")

    basicas = [v for v in vars_basicas]
    no_basicas = [v for v in visibles[1:-1] if v not in vars_basicas and not v.startswith('a')]
    lineas.append(f"- Variables básicas: {', '.join(basicas)}. Variables no básicas (valen 0): {', '.join(no_basicas) if no_basicas else 'ninguna'}.")

    if hay_art:
        lineas.append("- Las variables artificiales salieron de la base y valen 0: el problema original es factible.")

    fila_z = tabla[0, 1:9]
    alternativas = [columnas[j + 1] for j in range(8)
                    if columnas[j + 1] in visibles and columnas[j + 1] in no_basicas and abs(fila_z[j]) < 1e-5]
    if alternativas:
        lineas.append(f"- El coeficiente en la fila Z de {', '.join(alternativas)} es 0: pueden existir soluciones óptimas alternativas.")

    st.markdown("\n".join(lineas))


def metodo_simplex(opt, c, A, signos, b):
    st.subheader("Método Simplex / Penalización (M)")
    A, signos, b, cambios = preparar_restricciones(A, signos, b)
    for i in cambios:
        st.info(f"La restricción {i} tenía lado derecho negativo: se multiplicó por -1 (y se invirtió el signo) para que Lado derecho ≥ 0.")

    mostrar_sistema(opt, c, A, signos, b)
    visibles = columnas_visibles(signos)
    hay_art = any(s in (">=", "=") for s in signos)

    if hay_art:
        tabla_pre, columnas_pre, vb_pre = inicializar_simplex(opt, c, A, signos, b, ajustar=False)
        exp_pre = st.expander("Tabla simplex inicial (antes de ajustar la fila Z)")
        mostrar_tabla(exp_pre, tabla_pre, columnas_pre, vb_pre, visibles)

    tabla, columnas, vars_basicas = inicializar_simplex(opt, c, A, signos, b)
    iteracion = 0
    estado = None

    while True:
        # Expansor por iteración
        if iteracion == 0:
            titulo = "Iteración 0 (tabla inicial ajustada)" if hay_art else "Iteración 0 (tabla inicial)"
        else:
            titulo = f"Iteración {iteracion}"
        cont = st.expander(titulo, expanded=(iteracion == 0))
        mostrar_tabla(cont, tabla, columnas, vars_basicas, visibles)

        # Determinar variable de entrada
        fila_z = tabla[0, 1:9]
        if opt == "Max":
            if np.all(fila_z >= -1e-5):
                estado = "optimo"
                break
            col_entra = np.argmin(fila_z) + 1
        else:
            if np.all(fila_z <= 1e-5):
                estado = "optimo"
                break
            col_entra = np.argmax(fila_z) + 1

        # Determinar variable de salida (prueba del cociente)
        columna_pivote = tabla[1:, col_entra]
        rhs = tabla[1:, 9]

        cocientes = []
        for i in range(3):
            if columna_pivote[i] > 1e-5:
                cocientes.append(rhs[i] / columna_pivote[i])
            else:
                cocientes.append(np.inf)

        razones = pd.DataFrame({
            "Base": vars_basicas,
            "Lado derecho": rhs,
            f"Coef. de {columnas[col_entra]}": columna_pivote,
            "Razón": [f"{q:.4f}" if not np.isinf(q) else "—" for q in cocientes],
        })
        cont.markdown("**Razones (Lado derecho / coeficiente positivo de la columna de entrada):**")
        cont.dataframe(razones.style.format(precision=4, subset=["Lado derecho", f"Coef. de {columnas[col_entra]}"]), hide_index=True)

        if np.all(np.isinf(cocientes)):
            # Con una artificial positiva en la base, el problema penalizado no acotado no distingue
            # entre infactible y no acotado: se verifica si existe algún punto factible
            art_positiva = any(vb.startswith('a') and tabla[i+1, 9] > 1e-5 for i, vb in enumerate(vars_basicas))
            if art_positiva and not any(es_factible(p, A, signos, b) for p in calcular_intersecciones(A, b)):
                st.error("El problema es infactible: no existe ningún punto que cumpla todas las restricciones.")
                return {"estado": "infactible"}
            st.error("El problema es no acotado: ningún coeficiente de la columna de entrada es positivo.")
            return {"estado": "no acotado"}

        fila_sale = np.argmin(cocientes)
        pivote = tabla[fila_sale + 1, col_entra]

        cont.info(f"Variable de entrada: **{columnas[col_entra]}** | Variable de salida: **{vars_basicas[fila_sale]}** | Pivote: **{pivote:.2f}**")

        # Operaciones de Gauss-Jordan
        vars_basicas[fila_sale] = columnas[col_entra]
        tabla[fila_sale + 1, :] = tabla[fila_sale + 1, :] / pivote

        for i in range(4):
            if i != fila_sale + 1:
                factor = tabla[i, col_entra]
                tabla[i, :] = tabla[i, :] - factor * tabla[fila_sale + 1, :]

        iteracion += 1
        if iteracion > 15:
            st.warning("Se alcanzó el límite máximo de iteraciones (posible ciclo).")
            return {"estado": "limite"}

    # Una artificial positiva en la base indica infactibilidad
    for i, vb in enumerate(vars_basicas):
        if vb.startswith('a') and tabla[i+1, 9] > 1e-5:
            st.error("El problema es infactible: la solución óptima conserva una variable artificial en la base con valor > 0.")
            return {"estado": "infactible"}

    criterio = ">= 0" if opt == "Max" else "<= 0"
    st.success(f"Se ha alcanzado la solución óptima (todos los coeficientes de la fila Z son {criterio}).")

    # Resultados finales
    st.subheader("Resultados Finales del Modelo")
    z_opt = tabla[0, 9]

    def valor(var):
        if var in vars_basicas:
            return tabla[vars_basicas.index(var) + 1, 9]
        return 0.0

    st.metric("Valor Óptimo (Z)", f"{z_opt:.2f}")

    variables = [('x1', 'x1'), ('x2', 'x2')]
    for i in range(3):
        if signos[i] == "<=":
            variables.append((f's{i+1}', f's{i+1} (holgura)'))
        elif signos[i] == ">=":
            variables.append((f's{i+1}', f's{i+1} (superávit)'))
    columnas_res = st.columns(len(variables))
    for k, (var, etiqueta) in enumerate(variables):
        columnas_res[k].metric(etiqueta, f"{valor(var):.2f}")

    interpretar_simplex(opt, signos, cambios, tabla, columnas, vars_basicas, visibles, valor, z_opt, hay_art)
    return {"estado": "optimo", "x1": float(valor('x1')), "x2": float(valor('x2')), "Z": float(z_opt)}


def comparar_resultados(res_g, res_s):
    st.subheader("Comparación: método gráfico vs. simplex")
    if res_g is None or res_s is None:
        st.warning("No hay resultados para comparar.")
        return
    eg, es = res_g["estado"], res_s["estado"]
    if eg == "optimo" and es == "optimo":
        df = pd.DataFrame([
            {"Método": "Gráfico", "x1": res_g["x1"], "x2": res_g["x2"], "Z": res_g["Z"]},
            {"Método": "Simplex", "x1": res_s["x1"], "x2": res_s["x2"], "Z": res_s["Z"]},
        ])
        st.dataframe(df.round(4), hide_index=True)
        if abs(res_g["Z"] - res_s["Z"]) <= 1e-4 * max(1.0, abs(res_g["Z"])):
            st.success("Ambos métodos obtienen el mismo valor óptimo de Z.")
            if not np.allclose([res_g["x1"], res_g["x2"]], [res_s["x1"], res_s["x2"]], atol=1e-4):
                st.info("Los puntos difieren pero el valor de Z es el mismo: el problema tiene soluciones óptimas alternativas.")
        else:
            st.error("Los valores óptimos no coinciden.")
    elif eg == es and eg in ("infactible", "no acotado"):
        st.success(f"Ambos métodos coinciden: el problema es {eg}.")
    else:
        st.error(f"Los métodos no coinciden: gráfico = {eg}, simplex = {es}.")


# Interfaz principal
st.markdown(
    """
    <div class="encabezado">
        <h1>Tarea 2</h1>
        <p><b>IF-0017 Métodos Numéricos y Análisis Computacional</b>
        Anderson Jesús Monge Alvarado, C4H334<br>
        Jacqueline María Oviedo Miranda, C25748
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Datos de entrada
with st.sidebar:
    st.header("Ingreso de datos")

    with st.container(border=True):
        st.markdown("**Función objetivo**")
        opt = st.radio("Tipo de optimización", ["Maximización", "Minimización"], horizontal=True)
        col_c1, col_c2 = st.columns(2)
        c1 = col_c1.number_input("Coeficiente x1", value=1.0)
        c2 = col_c2.number_input("Coeficiente x2", value=1.0)

    A = np.zeros((3, 2))
    b = np.zeros(3)
    signos = []

    for i in range(3):
        with st.container(border=True):
            st.markdown(f"**Restricción {i+1}**")
            c_a1, c_a2 = st.columns(2)
            A[i][0] = c_a1.number_input(f"x1", value=1.0, key=f"a{i}1")
            A[i][1] = c_a2.number_input(f"x2", value=1.0, key=f"a{i}2")
            c_sig, c_b = st.columns(2)
            signos.append(c_sig.selectbox("Signo", ["<=", ">=", "="], key=f"s{i}"))
            b[i] = c_b.number_input(f"Lado derecho", value=10.0, key=f"b{i}")

    resolver = st.button("Resolver problema", type="primary")

# Modelo y resultados
with st.container(border=True):
    st.subheader("Modelo matemático")
    mostrar_modelo(opt, [c1, c2], A, signos, b)

if resolver:
    tab1, tab2, tab3 = st.tabs(["Método Gráfico", "Método Simplex", "Comparación"])

    with tab1:
        res_g = metodo_grafico(opt, [c1, c2], A, signos, b)

    with tab2:
        res_s = metodo_simplex(opt, [c1, c2], A, signos, b)

    with tab3:
        comparar_resultados(res_g, res_s)
else:
    st.info("Ingresa los datos en la barra lateral y presiona **Resolver problema**.")