# MetricLab Visual — núcleo del motor métrico

Sistema de búsqueda de imágenes por similitud y evaluación experimental de índices métricos.
Estructuras de Datos Avanzadas (SIS220) — Unidad 2, Métodos de Acceso Métrico.

Este repositorio es el **primer avance**: el motor C++ con el contrato común, los tres
índices del núcleo, el test oráculo y el arnés de benchmark. Todavía no hay descriptores
de imagen ni interfaz; se trabaja sobre datos sintéticos controlados, que es lo que
permite validar el motor antes de depender de Caltech-101.

---

## Compilar y ejecutar

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

./build/oracle_test        # verificación de correctitud
./build/bench              # mediciones -> resultados.csv
python3 scripts/plot_results.py
```

Requisitos: C++17, CMake ≥ 3.16. Para las gráficas, Python con matplotlib.

---

## Qué hay implementado

| Componente | Archivo | Papel |
|---|---|---|
| Contrato `MetricIndex` | `include/metriclab/index.hpp` | Interfaz que implementa todo índice |
| Métricas instrumentadas | `include/metriclab/metric.hpp` | Euclidiana, Manhattan, angular |
| Búsqueda secuencial | `include/metriclab/linear_scan.hpp` | Línea base y **oráculo** |
| VP-Tree | `src/vp_tree.cpp` | 4 políticas de pivote intercambiables |
| LAESA | `src/laesa.cpp` | Pivotes sin jerarquía |
| Traza JSON | `src/trace.cpp` | Puente con la visualización paso a paso |
| Test oráculo | `tests/oracle_test.cpp` | 10 503 comprobaciones |
| Benchmark | `bench/bench_main.cpp` | CSV listo para graficar |

### Tres decisiones de diseño que conviene defender en la exposición

1. **El contador de distancias vive dentro del objeto métrica**, no dentro de cada
   índice. Así ningún índice puede olvidarse de contar y las cifras son comparables por
   construcción. Un bug de conteo en un solo índice invalidaría toda la comparación.

2. **El test oráculo.** Todo índice exacto debe devolver el mismo resultado que la
   búsqueda secuencial. Sin esta prueba, un error en la condición de poda hace que un
   índice visite menos nodos y **parezca más rápido** cuando en realidad está perdiendo
   vecinos. Actualmente: 10 503 comprobaciones, 0 fallos.

3. **El contrato común va primero.** Añadir MVP-Tree, BST o M-tree significa escribir la
   clase y registrarla en una lista; el benchmark, el test y la visualización no se tocan.
   Esa es la razón por la que ampliar el estudio después resulta barato.

---

## Resultados del primer avance

Datos sintéticos, n = 5000, k = 10, 100 consultas de la misma distribución que los datos
pero fuera del índice. La métrica reportada es el **número de distancias calculadas por
consulta**, no el tiempo: el tiempo depende de la máquina y el compilador, el número de
distancias no.

### Hallazgo 1 — la dimensión ambiente no es lo que mata al índice

| dim | ρ | VP-Tree (max-var.) agrupado | VP-Tree (max-var.) uniforme |
|---|---|---|---|
| 4 | 4.0 / 5.0 | **93.0 %** podado | **92.3 %** podado |
| 16 | 11.4 / 22.0 | 80.9 % | 0.3 % |
| 64 | 19.8 / 90.8 | 79.6 % | 0.0 % |
| 256 | 24.6 / 363.3 | 80.3 % | 0.0 % |

Con datos **uniformes** la poda se derrumba entre d=4 y d=16 y a partir de d=64 el índice
no evita ni una sola comparación: es literalmente una búsqueda secuencial con sobrecoste.
Con datos **agrupados** la poda se estabiliza en torno al 80 % y **sigue ahí en dimensión
256**.

Conclusión provisional: lo que destruye la poda no es la dimensión del vector sino la
**ausencia de estructura** en los datos. Esto importa directamente para el proyecto,
porque los descriptores de Caltech-101 están agrupados por categoría: cabe esperar que
el VP-Tree siga siendo útil en dimensión alta, al contrario de lo que sugiere la
formulación ingenua de la "maldición de la dimensionalidad".

### Hallazgo 2 — ρ por sí sola no predice el rendimiento

En ρ ≈ 22 hay dos casos con comportamiento opuesto: uniforme d=16 (ρ=22.0, poda 0.3 %) y
agrupado d=256 (ρ=24.6, poda 80.3 %). La razón es que ρ se calcula sobre pares al azar,
y en datos agrupados ese histograma está dominado por pares **entre** grupos, mientras
que la búsqueda se juega **dentro** del grupo de la consulta.

Esto refina la hipótesis de partida y es material directo para el informe: habrá que
medir ρ **local** (sobre los vecinos de la consulta) y no solo global.

### Hallazgo 3 — la política de pivote vale hasta 19 puntos

Con datos agrupados, d=32, n=5000:

| política | distancias/consulta | poda |
|---|---|---|
| max-varianza (Yianilos) | 1036 | **79.3 %** |
| max-rango | 1180 | 76.4 % |
| aleatoria | 1227 | 75.5 % |
| FFT | 1973 | 60.5 % |

El mismo algoritmo, los mismos datos, las mismas consultas: solo cambia cómo se elige el
pivote. Max-varianza gana de forma consistente en todas las dimensiones probadas y FFT
pierde siempre. Recordatorio metodológico: cada política **construye un árbol distinto**,
no reconfigura el mismo.

### Hallazgo 4 — LAESA gana al VP-Tree en todos los casos probados

Entre 2 y 5 veces menos distancias por consulta. Esto justifica haberlo incluido: si solo
hubiéramos comparado VP-Tree contra la búsqueda secuencial, habríamos concluido que la
jerarquía es la causa de la mejora, cuando una tabla de pivotes sin ningún árbol lo hace
mejor.

El contrapeso hay que reportarlo: LAESA gasta **O(n·m) distancias al construirse** y
guarda una tabla n×m en memoria. Es la relación memoria/precálculo contra consulta, y es
un resultado más interesante que "el árbol gana".

### Hallazgo 5 — la ventaja crece con n

Datos agrupados, d=32: la poda del VP-Tree pasa de 52.6 % con n=1000 a 81.5 % con
n=10000. El número absoluto de distancias crece mucho más despacio que n, que es
exactamente el comportamiento sublineal que se espera de un índice que funciona.

---

## Advertencia metodológica que ya costó un resultado falso

La primera versión del benchmark usaba **consultas uniformes sobre datos agrupados**.
Con esa configuración la poda parecía derrumbarse al 0 % en dimensión 128 incluso con
datos agrupados, y la conclusión habría sido la contraria a la correcta.

El motivo: si la consulta no pertenece a la distribución de los datos, cae lejos de todo,
el radio dinámico τ nunca se cierra y no se puede podar. Las consultas deben venir de la
misma distribución que la colección, pero no estar dentro del índice (una consulta que ya
está indexada tiene un vecino a distancia 0 y sesga el resultado).

Vale la pena dejar esto escrito en el informe: es una decisión experimental que cambia
las conclusiones.

---

## Siguientes pasos

**Inmediato**
1. Extracción de descriptores de Caltech-101 en Python (histograma HSV + opcionalmente
   embeddings de una CNN preentrenada), volcado a formato binario que lea el motor.
2. Enlace pybind11 para llamar al motor desde Python sin pasar por archivos temporales.
3. Precision@k con consultas estratificadas por clase: Caltech-101 está muy
   desbalanceado y un muestreo al azar mide sobre todo las clases grandes.

**Después**
4. Backend FastAPI que exponga búsqueda y traza.
5. Interfaz React: buscador, reproductor paso a paso de la traza, laboratorio de
   experimentos.

**Ampliaciones condicionadas** — se añaden solo si responden una pregunta que el núcleo
no responde, no por aumentar el alcance:
- **MVP-Tree**, si queremos medir cuánta poda se consigue reutilizando distancias ya
  calculadas, sin calcular ninguna nueva.
- **BST**, si queremos saber si la geometría de partición (bisector contra radio) importa.
- **M-tree**, si decidimos estudiar inserciones y borrados.

El contrato `MetricIndex` ya está hecho para que cualquiera de las tres cueste poco.

---

## Estructura

```
metriclab/
├── include/metriclab/
│   ├── metric.hpp        # distancias + contador
│   ├── dataset.hpp       # colección de vectores
│   ├── index.hpp         # CONTRATO + traza
│   ├── linear_scan.hpp   # oráculo
│   ├── vp_tree.hpp
│   ├── laesa.hpp
│   └── synthetic.hpp     # datos de prueba, ρ, validación de métrica
├── src/                  # vp_tree.cpp, laesa.cpp, trace.cpp
├── tests/oracle_test.cpp
├── bench/bench_main.cpp
├── scripts/plot_results.py
└── CMakeLists.txt
```
