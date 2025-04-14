import numpy as np
import matplotlib.pyplot as plt
from Inventory_Model import Inventario_de_tienda
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from bootstrap import analisis_bootstrap_politicas, graficar_resultados_bootstrap, prueba_hipotesis
from common_functions import funcion_costo

# Definir funciones de distribución a nivel de módulo (no dentro del main)
def demanda_normal():
    return max(0, round(np.random.normal(3, 1)))
    
def demanda_exponencial():
    return round(np.random.exponential(3))

def simular_punto(params):
    """Función para simular un punto (s,S) específico y retornar el beneficio"""
    s, S, λ, r, h, L, T, G = params
    if s >= S:  # Restricción obligatoria
        return None
    
    sim = Inventario_de_tienda(
        r=r, λ=λ, h=h, L=L, T=T, s=s, S=S,
        c=funcion_costo,  # Usar función normal en lugar de lambda
        G=G
    )
    res = sim.Simular()
    return res["Beneficio Total"]

def simular_s_S_optimo(G_distribuciones, lambda_fijo=3, 
                       valores_S=None, valores_s=None,
                       T=10000, r=100, h=0.5, L=1.0,
                        max_workers=None):
    """
    Simula valores óptimos de s y S para un valor fijo de λ
    Usa un paso (step) para reducir la cantidad de puntos evaluados
    y aprovecha paralelismo para acelerar la ejecución
    """
    # Usar valores por defecto si no se especifican
    if valores_S is None:
        valores_S = list(range(55, 1001))  # Usa paso para reducir puntos
    if valores_s is None:
        valores_s = list(range(5, 1001))   # Usa paso para reducir puntos
        
    resultados_por_distribucion = {}
    λ = lambda_fijo
    
    print(f"Simulando con λ = {λ} (tasa de llegada de clientes)")
    print(f"Evaluando {len(valores_s)} valores de s y {len(valores_S)} valores de S")
    print(f"Usando {multiprocessing.cpu_count()} CPUs")
    
    for nombre, G in G_distribuciones.items():
        print(f"Distribución: {nombre}")
        resultados_S = {}
        
        for S in valores_S:
            print(f"  Simulando con S = {S}")
            
            # Crear lista de tareas para ejecución en paralelo
            tareas = []
            for s in valores_s:
                if s < S:  # Solo incluir combinaciones válidas
                    tareas.append((s, S, λ, r, h, L, T, G))
            
            # Ejecutar tareas en paralelo
            beneficios = []
            s_validos = []
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                resultados = list(executor.map(simular_punto, tareas))
                
                for i, beneficio in enumerate(resultados):
                    if beneficio is not None:
                        s_validos.append(tareas[i][0])  # s
                        beneficios.append(beneficio)
            
            # Almacenar resultados para cada S
            resultados_S[S] = {
                "s_values": s_validos,
                "beneficios": beneficios,
                "s_optimo": s_validos[np.argmax(beneficios)] if beneficios else None,
                "beneficio_max": np.max(beneficios) if beneficios else None
            }
        
        resultados_por_distribucion[nombre] = resultados_S
    
    return resultados_por_distribucion

def graficar_resultados_s_S(resultados_por_distribucion, G_distribuciones, valores_S, lambda_fijo=5):
    """
    Genera una gráfica limpia que muestra la relación entre el beneficio y 
    el punto de reorden (s) para diferentes niveles máximos de inventario (S),
    mostrando los valores óptimos solo en una tabla.
    """
    # Configurar el tamaño de la figura
    plt.figure(figsize=(14, 10))
    
    # Crear subplots para cada distribución
    for i, dist_nombre in enumerate(G_distribuciones.keys()):
        plt.subplot(len(G_distribuciones), 1, i+1)
        
        # Lista para almacenar los puntos óptimos (solo para la tabla)
        s_optimos = []
        beneficios_max = []
        S_valores = []
        
        # Colores para diferentes valores de S
        colores = plt.cm.viridis(np.linspace(0, 1, len(valores_S)))
        
        # Graficar curvas para cada S - usar líneas más delgadas para no sobrecargar visual
        for j, S in enumerate(valores_S):
            if S in resultados_por_distribucion[dist_nombre]:
                datos = resultados_por_distribucion[dist_nombre][S]
                
                if datos["s_values"] and datos["beneficios"]:
                    # Graficar la curva de beneficio vs s (líneas más delgadas)
                    plt.plot(datos["s_values"], datos["beneficios"], 
                            marker='.', markersize=2, linestyle='-', linewidth=1.0,
                            color=colores[j], label=f'S = {S}', alpha=0.7)
                    
                    # Almacenar punto óptimo si existe (solo para la tabla)
                    if datos["s_optimo"] is not None:
                        s_optimos.append(datos["s_optimo"])
                        beneficios_max.append(datos["beneficio_max"])
                        S_valores.append(S)
        
        # Zona para tabular los puntos óptimos
        # Crear una tabla dentro de la gráfica con los valores óptimos
        if s_optimos:
            # Ordenar por beneficio descendente
            datos_tabla = sorted(zip(S_valores, s_optimos, beneficios_max), 
                               key=lambda x: x[2], reverse=True)
            
            # Preparar datos para la tabla
            tabla_datos = [["S", "s", "Beneficio"]]
            for S, s, b in datos_tabla[:5]:  # Mostrar solo los 5 mejores
                tabla_datos.append([f"{S}", f"{int(s)}", f"{b:.2f}"])
            
            # Añadir la tabla - posición en la esquina superior derecha
            tabla = plt.table(cellText=tabla_datos, 
                             loc='upper right', 
                             cellLoc='center',
                             bbox=[0.7, 0.55, 0.25, 0.35])  # Posición y tamaño
            
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(10)
            tabla.scale(1, 1.5)  # Ajustar altura de filas
            
            # Destacar la primera fila (encabezado) y la segunda (mejor valor)
            for (row, col), cell in tabla.get_celld().items():
                if row == 0:  # Encabezado
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor('#cccccc')
                elif row == 1:  # Mejor valor
                    cell.set_facecolor('#ffffcc')
        
        # Ajustar la gráfica
        plt.title(f"Distribución {dist_nombre}: Beneficio vs Punto de reorden (λ={lambda_fijo})", fontsize=15)
        plt.xlabel("s (punto de reorden)", fontsize=12)
        plt.ylabel("Beneficio promedio ((R-C-H)/T)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Agregar leyenda con los valores relevantes de S
        if len(valores_S) > 10:
            # Si hay muchos valores de S, mostrar solo algunos en la leyenda
            indices = np.linspace(0, len(valores_S)-1, min(8, len(valores_S))).astype(int)
            handles, labels = plt.gca().get_legend_handles_labels()
            
            # Filtrar
            filtered_handles = [handles[j] for j in indices if j < len(handles)]
            filtered_labels = [labels[j] for j in indices if j < len(labels)]
            
            plt.legend(filtered_handles, filtered_labels, loc='upper left', fontsize=10)
        else:
            plt.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar la combinación óptima global con formato mejorado
    mejor_dist = None
    mejor_S = None
    mejor_s = None
    mejor_beneficio = -float('inf')
    
    for dist_nombre, resultados_S in resultados_por_distribucion.items():
        for S, datos in resultados_S.items():
            if datos["beneficio_max"] is not None and datos["beneficio_max"] > mejor_beneficio:
                mejor_beneficio = datos["beneficio_max"]
                mejor_dist = dist_nombre
                mejor_S = S
                mejor_s = datos["s_optimo"]
    
    # Mostrar resultados como tabla
    print("\n" + "="*50)
    print(" 🏆 COMBINACIÓN ÓPTIMA GLOBAL 🏆 ".center(50))
    print("="*50)
    print(f"│ Distribución: {mejor_dist}")
    print(f"│ S óptimo: {mejor_S}")
    print(f"│ s óptimo: {int(mejor_s)}")
    print(f"│ Beneficio máximo: {mejor_beneficio:.2f}")
    print("="*50)


# Poner todo el código de ejecución bajo este bloque
if __name__ == '__main__':
    # Usar las funciones ya definidas a nivel de módulo
    G_distribuciones = {
        "Normal": demanda_normal,
        "Exponencial": demanda_exponencial
    }

    # Primera ejecución: barrido rápido para encontrar políticas candidatas
    valores_s = list(range(5, 501, 20))  # Paso mayor para ejecutar más rápido
    valores_S = list(range(50, 511, 20))  # Paso mayor para ejecutar más rápido

    resultados = simular_s_S_optimo(
        G_distribuciones, 
        lambda_fijo=3,
        valores_s=valores_s,
        valores_S=valores_S,
        T=10000,     
        max_workers=None
    )

    # Mostrar los resultados iniciales
    graficar_resultados_s_S(resultados, G_distribuciones, valores_S, lambda_fijo=3)
    
    # Definir las distribuciones de demanda que usarás
    G_distribuciones = {
        "Normal": demanda_normal,
        "Exponencial": demanda_exponencial
    }
    
    # Extraer las políticas óptimas del diccionario de resultados
    politicas_candidatas = []
    for dist_nombre, resultados_S in resultados.items():
        # Para cada distribución, encontrar la mejor política (s, S)
        mejor_S = None
        mejor_s = None
        mejor_beneficio = -float('inf')
        
        for S, datos in resultados_S.items():
            if datos["beneficio_max"] is not None and datos["beneficio_max"] > mejor_beneficio:
                mejor_beneficio = datos["beneficio_max"]
                mejor_S = S
                mejor_s = datos["s_optimo"]
        
        if mejor_S is not None and mejor_s is not None:
            politicas_candidatas.append((mejor_s, mejor_S))
    
    print("Políticas candidatas seleccionadas para bootstrap:")
    for i, (s, S) in enumerate(politicas_candidatas):
        print(f"Política {i+1}: s={s}, S={S}")
    
    # Realizar el análisis bootstrap para cada política candidata y para cada distribución
    bootstrap_results = analisis_bootstrap_politicas(
        G_distribuciones=G_distribuciones,
        politicas_candidatas=politicas_candidatas,
        lambda_fijo=3,
        T=10000,
        n_replicas=30,
        n_bootstrap=1000
    )
    
    # Visualizar resultados bootstrap para cada distribución
    for dist in G_distribuciones.keys():
        graficar_resultados_bootstrap(bootstrap_results, dist)
        prueba_hipotesis(bootstrap_results, dist)