import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Inventory_Model import Inventario_de_tienda
from common_functions import funcion_costo

# --- Funciones de bootstrap ---

def analisis_bootstrap_politicas(G_distribuciones, politicas_candidatas, lambda_fijo=3, T=10000, n_replicas=30, n_bootstrap=1000):
    """
    Para cada política candidata (s, S) y para cada distribución en G_distribuciones,
    ejecuta n_replicas simulaciones para obtener una muestra del beneficio por unidad de tiempo,
    y realiza n_bootstrap muestras bootstrap para obtener la distribución de las medias.
    
    Devuelve un diccionario con, para cada política, la distribución de las medias, la media bootstrap,
    el error estándar y el intervalo de confianza del 95%.
    """
    bootstrap_results = {}
    
    for dist_name, G in G_distribuciones.items():
        bootstrap_results[dist_name] = {}
        for policy in politicas_candidatas:
            s, S = policy
            # Obtener resultados de n_replicas para la política candidata
            profits = []
            for i in range(n_replicas):
                sim = Inventario_de_tienda(
                    r=100,
                    λ=lambda_fijo,
                    h=0.5,
                    L=1.0,
                    T=T,
                    s=s,
                    S=S,
                    c=funcion_costo,
                    G=G
                )
                res = sim.Simular()
                # Tomamos la ganancia por unidad de tiempo como variable de interés
                profits.append(res["Beneficio por Unidad de Tiempo"])
            
            # Realizar bootstrap: generar n_bootstrap réplicas de la media
            bootstrap_means = []
            for j in range(n_bootstrap):
                muestra_boot = np.random.choice(profits, size=len(profits), replace=True)
                bootstrap_means.append(np.mean(muestra_boot))
            bootstrap_means = np.array(bootstrap_means)
            
            # Estadísticos del bootstrap
            mean_bootstrap = np.mean(bootstrap_means)
            std_bootstrap  = np.std(bootstrap_means)
            lower_ci = np.percentile(bootstrap_means, 2.5)
            upper_ci = np.percentile(bootstrap_means, 97.5)
            
            bootstrap_results[dist_name][policy] = {
                "bootstrap_means": bootstrap_means,
                "mean_bootstrap": mean_bootstrap,
                "std_bootstrap": std_bootstrap,
                "95ci": (lower_ci, upper_ci),
                "original_profits": profits
            }
    return bootstrap_results

def graficar_resultados_bootstrap(bootstrap_results, dist_name):
    """
    Para cada política candidata en una distribución determinada, traza un histograma
    de la distribución de las medias bootstrap, marcando la media y los límites del intervalo de confianza.
    """
    for policy, results in bootstrap_results[dist_name].items():
        s, S = policy
        plt.figure(figsize=(8, 5))
        sns.histplot(results["bootstrap_means"], kde=True, bins=30)
        plt.title(f"Distribución bootstrap para política s={s}, S={S} \n(distr. {dist_name})", fontsize=14)
        plt.xlabel("Beneficio por Unidad de Tiempo", fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        plt.axvline(results["mean_bootstrap"], color='red', linestyle='--', label=f"Media Bootstrap: {results['mean_bootstrap']:.2f}")
        plt.axvline(results["95ci"][0], color='green', linestyle='--', label=f"2.5%: {results['95ci'][0]:.2f}")
        plt.axvline(results["95ci"][1], color='green', linestyle='--', label=f"97.5%: {results['95ci'][1]:.2f}")
        plt.legend()
        plt.show()

def prueba_hipotesis(bootstrap_results, dist_name):
    """
    Compara las políticas candidatas para la distribución indicada, identifica la política con mayor
    media bootstrap y muestra los intervalos de confianza de todas las políticas para determinar
    si la diferencia es estadísticamente significativa.
    """
    # Buscar la política ganadora (la de mayor media bootstrap)
    best_policy = None
    best_mean = -np.inf
    for policy, results in bootstrap_results[dist_name].items():
        if results["mean_bootstrap"] > best_mean:
            best_mean = results["mean_bootstrap"]
            best_policy = policy
            
    print(f"\nPara la distribución {dist_name}:")
    print(f"➡ La mejor política es s = {best_policy[0]}, S = {best_policy[1]} con una media bootstrap de {best_mean:.2f}\n")
    print("Intervalos de confianza de las políticas:")
    for policy, results in bootstrap_results[dist_name].items():
        ci = results["95ci"]
        print(f"Política s = {policy[0]}, S = {policy[1]} -> IC 95%: {ci[0]:.2f} a {ci[1]:.2f}")
    
    # Prueba de hipótesis: comparar la política ganadora con cada otra política
    best_ci_lower, best_ci_upper = bootstrap_results[dist_name][best_policy]["95ci"]
    print("\nPrueba de hipótesis (comparación entre políticas):")
    for policy, results in bootstrap_results[dist_name].items():
        if policy == best_policy:
            continue
        other_ci_lower, other_ci_upper = results["95ci"]
        if best_ci_lower > other_ci_upper:
            print(f"La política s = {best_policy[0]}, S = {best_policy[1]} es significativamente mejor que la política s = {policy[0]}, S = {policy[1]}.")
        else:
            print(f"La diferencia entre la política s = {best_policy[0]}, S = {best_policy[1]} y la política s = {policy[0]}, S = {policy[1]} NO es significativa al 95%.")
