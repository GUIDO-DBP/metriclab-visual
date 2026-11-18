// metriclab/vp_tree.hpp
//
// VP-Tree (Vantage Point Tree, Yianilos 1993) con POLITICA DE PIVOTE
// INTERCAMBIABLE.
//
// La politica es el objeto central del estudio. Nota metodologica importante:
// cambiar la politica construye un ARBOL DISTINTO, no reconfigura el mismo.
// Por eso el experimento mantiene fijos los datos, la metrica y el conjunto
// de consultas, y varia unicamente la estrategia de construccion.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#ifndef METRICLAB_VP_TREE_HPP
#define METRICLAB_VP_TREE_HPP

#include "metriclab/index.hpp"

#include <random>
#include <string>
#include <vector>

namespace metriclab {

enum class PivotPolicy {
    Random,       // linea base: pivote al azar
    MaxVariance,  // Yianilos: maximiza la varianza de distancias a una muestra
    MaxSpread,    // maximiza (max - min) de esas distancias
    FarthestFirst // el mas alejado de un objeto de referencia (FFT)
};

std::string pivotPolicyName(PivotPolicy p);

struct VpTreeConfig {
    PivotPolicy policy = PivotPolicy::MaxVariance;
    std::size_t leaf_size = 1;       // 1 = VP-Tree clasico
    std::size_t sample_candidates = 10;  // candidatos a pivote evaluados
    std::size_t sample_test = 25;    // muestra contra la que se evalua cada uno
    unsigned seed = 42;              // reproducibilidad de los experimentos
};

class VpTree : public MetricIndex {
public:
    explicit VpTree(VpTreeConfig cfg = {}) : cfg_(cfg), rng_(cfg.seed) {}

    void build(const Dataset& data, const Metric& metric) override;

    std::vector<Neighbor> searchKnn(const float* q, std::size_t k,
                                    Trace* trace = nullptr) const override;
    std::vector<Neighbor> searchRange(const float* q, double radius,
                                      Trace* trace = nullptr) const override;

    std::string name() const override {
        return "VP-Tree[" + pivotPolicyName(cfg_.policy) + "]";
    }
    std::size_t memoryBytes() const override;

    // Metricas de calidad de la estructura, no de velocidad.
    std::size_t nodeCount() const { return nodes_.size(); }
    int height() const;
    // Topologia real para el visualizador web; no modifica la busqueda.
    void writeStructure(std::ostream& os) const;

private:
    struct Node {
        ObjectId pivot = 0;
        double mu = 0.0;          // radio de corte (mediana)
        int left = -1;            // d(pivote, x) <  mu
        int right = -1;           // d(pivote, x) >= mu
        bool leaf = false;
        std::vector<ObjectId> bucket;  // solo en hojas
    };

    int buildNode(std::vector<ObjectId>& ids, int depth);
    std::size_t choosePivot(const std::vector<ObjectId>& ids);
    int heightFrom(int node) const;

    VpTreeConfig cfg_;
    mutable std::mt19937 rng_;
    std::vector<Node> nodes_;
    int root_ = -1;
    const Dataset* data_ = nullptr;
    const Metric* metric_ = nullptr;
};

}  // namespace metriclab

#endif  // METRICLAB_VP_TREE_HPP
