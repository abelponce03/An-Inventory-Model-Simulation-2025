import numpy as np

class Inventario_de_tienda:
    """
    Simula un sistema de inventario con política (s, S) para una tienda.
    """
  
    def __init__(self, r, λ, h, L, T, s, S, c, G):
        # Validación de parámetros
        if s >= S:
            raise ValueError("El punto de reorden (s) debe ser menor que el nivel máximo de inventario (S)")
        
        # Parámetros del sistema
        self.r = r               # Precio de venta por unidad
        self.λ = λ               # Tasa de llegada de clientes (Poisson)
        self.h = h               # Costo de mantenimiento por unidad/tiempo
        self.L = L               # Tiempo de entrega de pedidos
        self.T = T               # Tiempo total de simulación
        self.s = s               # Punto de reorden
        self.S = S               # Nivel máximo de inventario
        self.c = c               # Función de costo de pedido c(y)
        self.G = G              # Función de distribución de demanda

        # Estado inicial
        self.t = 0.0             # Tiempo actual
        self.x = S               # Inventario actual (x)
        self.y = 0              # Pedido pendiente (y)
        self.C = 0.0             # Costo total de pedidos
        self.H = 0.0             # Costo total de mantenimiento
        self.R = 0.0             # Ingresos totales

        # Eventos futuros
        self.t0 = self.generar_proximo_cliente()  # Próxima llegada de cliente
        self.t1 = float('inf')                    # Próxima entrega de pedido

    def generar_proximo_cliente(self):
        """Genera el tiempo hasta la próxima llegada de cliente usando -log(U)/λ."""
        U = np.random.uniform(0, 1)
        return self.t - (np.log(U) / self.λ)

    def Simular(self):
        """Ejecuta la simulación hasta el tiempo T."""
        while self.t < self.T:
            if self.t0 < self.t1:
                self.manejar_llegada_de_clientes()
            else:
                self.manejar_pedidos_de_entrega()
        return self.resultados()

    def manejar_llegada_de_clientes(self):
        """Procesa la llegada de un cliente (Caso 1)."""
        # Actualizar costos de mantenimiento
        self.H += (self.t0 - self.t) * self.x * self.h
        self.t = self.t0

        # Generar demanda y vender
        D = self.G()                  # Demanda del cliente
        w = min(D, self.x)            # Unidades vendidas
        self.R += w * self.r          # Actualizar ingresos
        self.x -= w                   # Reducir inventario

        # Hacer pedido si es necesario (y no hay pedidos pendientes)
        if self.x < self.s and self.y == 0:
            self.y = self.S - self.x
            self.t1 = self.t + self.L
            self.C += self.c(self.y)  # Añadir costo del pedido

        # Programar próxima llegada de cliente
        self.t0 = self.generar_proximo_cliente()

    def manejar_pedidos_de_entrega(self):
        """Procesa la entrega de un pedido (Caso 2)."""
        # Actualizar costos de mantenimiento
        self.H += (self.t1 - self.t) * self.x * self.h
        self.t = self.t1

        # Recibir pedido y reiniciar variables
        self.x += self.y
        self.y = 0
        self.t1 = float('inf')

    def resultados(self):
        """Devuelve un resumen de los resultados."""
        profit = self.R - self.C - self.H
        return {
            "Ingresos Totales (R)": self.R,
            "Costo de Pedidos (C)": self.C,
            "Costo de Mantenimiento (H)": self.H,
            "Beneficio Total": profit,
            "Beneficio por Unidad de Tiempo": profit / self.T
        }

# Ejemplo de uso con demanda Poisson (λ=3)
if __name__ == "__main__":
    # Función de costo de pedido c(y) = 50 + 2y
    costo_pedido = lambda y: 50 + 2 * y
    
    # Función de demanda Poisson (λ=3)
    demanda_poisson = lambda: np.random.poisson(3)

    # Configurar simulación
    sim = Inventario_de_tienda(
        r=10,
        λ=2,
        h=0.5,
        L=1.0,
        T=1000,
        s=10,
        S=50,
        c=costo_pedido,
        G=demanda_poisson
    )

    resultados = sim.Simular()
    for clave, valor in resultados.items():
        print(f"{clave}: {valor:.2f}")