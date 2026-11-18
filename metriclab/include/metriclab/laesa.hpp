// metriclab/laesa.hpp
//
// LAESA: indexacion por TABLA DE PIVOTES, sin jerarquia.
//
// Papel en el proyecto (y la razon de que no sea relleno): VP-Tree combina
// dos ideas a la vez, usar pivotes y organizar los datos en un arbol. Si solo
// comparamos VP-Tree contra la busqueda secuencial y gana, no sabemos cual de
// las dos ideas produjo la ganancia. LAESA usa pivotes SIN arbol, asi que
// separa las dos contribuciones.
//
// Matiz honesto que debe constar en el informe: LAESA y VP-Tree no difieren
// en una sola variable. Tambien difieren en precalculo (LAESA guarda una
// tabla n x m) y en estrategia de consulta. Lo que comparamos son mecanismos
// completos, no una variable aislada, y asi hay que redactarlo.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#ifndef METRICLAB_LAESA_HPP
#define METRICLAB_LAESA_HPP

#include "metriclab/index.hpp"

#include <random>
#include <vector>

namespace metriclab {

struct LaesaConfig {
    std::size_t num_pivots = 16;
    unsigned seed = 42;
};

class Laesa : public MetricIndex {
public:
    explicit Laesa(LaesaConfig cfg = {}) : cfg_(cfg) {}

    void build(const Dataset& data, const Metric& metric) override;

    std::vector<Neighbor> searchKnn(const float* q, std::size_t k,
                                    Trace* trace = nullptr) const override;
    std::vector<Neighbor> searchRange(const float* q, double radius,
                                      Trace* trace = nullptr) const override;

    std::string name() const override {
        return "LAESA[m=" + std::to_string(cfg_.num_pivots) + "]";
    }
    std::size_t memoryBytes() const override;

    const std::vector<ObjectId>& pivots() const { return pivots_; }

private:
    // table_[i * m + j] = d(objeto_i, pivote_j)
    double tableAt(ObjectId i, std::size_t j) const {
        return table_[i * pivots_.size() + j];
    }

    LaesaConfig cfg_;
    std::vector<ObjectId> pivots_;
    std::vector<double> table_;
    const Dataset* data_ = nullptr;
    const Metric* metric_ = nullptr;
};

}  // namespace metriclab

#endif  // METRICLAB_LAESA_HPP
