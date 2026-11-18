// metriclab/linear_scan.hpp
//
// Busqueda secuencial. Doble papel en el proyecto:
//   1) linea base de rendimiento (n distancias por consulta, siempre);
//   2) ORACULO de correccion: todo indice exacto debe devolver exactamente
//      este resultado. Sin esta comprobacion, un bug de poda hace que un
//      indice parezca mas rapido en vez de parecer roto.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#ifndef METRICLAB_LINEAR_SCAN_HPP
#define METRICLAB_LINEAR_SCAN_HPP

#include "metriclab/index.hpp"

#include <algorithm>

namespace metriclab {

class LinearScan : public MetricIndex {
public:
    void build(const Dataset& data, const Metric& metric) override {
        data_ = &data;
        metric_ = &metric;
        build_distances_ = 0;  // la busqueda secuencial no construye nada
    }

    std::vector<Neighbor> searchKnn(const float* q, std::size_t k,
                                    Trace* trace = nullptr) const override {
        std::vector<Neighbor> all;
        all.reserve(data_->size());
        for (ObjectId i = 0; i < data_->size(); ++i) {
            const double d = (*metric_)(q, data_->at(i));
            if (trace) trace->add({EventKind::ComputeDist, -1, (long long)i, d, 0, 0, 0, 0,
                                   "comparacion directa"});
            all.push_back({i, d});
        }
        const std::size_t kk = std::min(k, all.size());
        std::partial_sort(all.begin(), all.begin() + kk, all.end());
        all.resize(kk);
        return all;
    }

    std::vector<Neighbor> searchRange(const float* q, double radius,
                                      Trace* trace = nullptr) const override {
        std::vector<Neighbor> out;
        for (ObjectId i = 0; i < data_->size(); ++i) {
            const double d = (*metric_)(q, data_->at(i));
            if (trace) trace->add({EventKind::ComputeDist, -1, (long long)i, d, radius, 0, 0, 0,
                                   "comparacion directa"});
            if (d <= radius) out.push_back({i, d});
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    std::string name() const override { return "LinearScan"; }
    std::size_t memoryBytes() const override { return 0; }

private:
    const Dataset* data_ = nullptr;
    const Metric* metric_ = nullptr;
};

}  // namespace metriclab

#endif  // METRICLAB_LINEAR_SCAN_HPP
