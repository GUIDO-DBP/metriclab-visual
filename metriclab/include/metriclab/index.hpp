// metriclab/index.hpp
//
// EL CONTRATO. Todo indice metrico del proyecto implementa esta interfaz.
//
// Por que esto va primero, antes que cualquier algoritmo:
//   - Anadir una estructura nueva (MVP-Tree, BST, M-tree) no obliga a tocar
//     el benchmark, el test oraculo ni la visualizacion.
//   - El arnes de medicion recorre indices por puntero a la base; no sabe
//     ni le importa cual es cual.
//   - La traza tiene un formato unico, asi que la interfaz web dibuja lo que
//     el motor reporto y nunca reimplementa logica de poda.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#ifndef METRICLAB_INDEX_HPP
#define METRICLAB_INDEX_HPP

#include "metriclab/dataset.hpp"
#include "metriclab/metric.hpp"

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace metriclab {

struct Neighbor {
    ObjectId id;
    double dist;
    bool operator<(const Neighbor& o) const {
        // Orden por distancia; desempate por id para que el resultado sea
        // determinista y el test oraculo pueda comparar listas exactas.
        if (dist != o.dist) return dist < o.dist;
        return id < o.id;
    }
};

// ---------------------------------------------------------------------------
// Traza: registro de lo que hizo el algoritmo, para la pantalla paso a paso.
// ---------------------------------------------------------------------------

enum class EventKind {
    VisitNode,     // se entra a un nodo del indice
    ComputeDist,   // se calcula una distancia real contra un objeto
    Prune,         // se descarta un subarbol o un candidato
    Candidate,     // un objeto entra al conjunto de los k mejores
    Descend        // se elige un subarbol para seguir
};

struct TraceEvent {
    EventKind kind;
    int node = -1;            // id de nodo (-1 si no aplica)
    long long object = -1;    // id de objeto (-1 si no aplica)
    double dist = 0.0;        // distancia calculada
    double bound = 0.0;       // cota inferior o superior usada en la decision
    double radius = 0.0;      // radio de cobertura / mu del nodo
    double tau = 0.0;         // radio dinamico de la busqueda en ese instante
    int depth = 0;
    std::string reason;       // explicacion legible, se muestra en la interfaz
};

// Se pasa un puntero opcional a las busquedas. Si es nullptr, no se traza y
// el costo es cero: las mediciones de tiempo del benchmark no quedan
// contaminadas por la instrumentacion de la demo.
class Trace {
public:
    void add(TraceEvent e) { events_.push_back(std::move(e)); }
    const std::vector<TraceEvent>& events() const { return events_; }
    void clear() { events_.clear(); }
    std::size_t size() const { return events_.size(); }

    // Serializa a JSON para que la interfaz web lo consuma tal cual.
    void writeJson(std::ostream& os) const;

private:
    std::vector<TraceEvent> events_;
};

// ---------------------------------------------------------------------------
// Interfaz comun
// ---------------------------------------------------------------------------

class MetricIndex {
public:
    virtual ~MetricIndex() = default;

    // El indice guarda referencias a dataset y metrica; ambos deben
    // sobrevivir al indice.
    virtual void build(const Dataset& data, const Metric& metric) = 0;

    // k vecinos mas cercanos, ordenados por distancia creciente.
    virtual std::vector<Neighbor> searchKnn(const float* q, std::size_t k,
                                            Trace* trace = nullptr) const = 0;

    // Todos los objetos a distancia <= radius.
    virtual std::vector<Neighbor> searchRange(const float* q, double radius,
                                              Trace* trace = nullptr) const = 0;

    virtual std::string name() const = 0;

    // Memoria de la ESTRUCTURA, sin contar los vectores del dataset.
    virtual std::size_t memoryBytes() const = 0;

    // Distancias gastadas durante la construccion. Es un costo real que
    // los informes suelen omitir y que cambia las conclusiones cuando el
    // indice se usa pocas veces.
    std::size_t buildDistances() const { return build_distances_; }

protected:
    std::size_t build_distances_ = 0;
};

}  // namespace metriclab

#endif  // METRICLAB_INDEX_HPP
