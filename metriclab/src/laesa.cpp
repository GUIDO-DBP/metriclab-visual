// metriclab/src/laesa.cpp
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#include "metriclab/laesa.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>

namespace metriclab {

// Seleccion de pivotes por FFT (farthest-first traversal): se empieza con uno
// al azar y cada nuevo pivote es el objeto que maximiza su distancia MINIMA a
// los ya elegidos. Busca pivotes repartidos por todo el espacio, porque dos
// pivotes cercanos dan cotas casi identicas y uno de los dos no aporta nada.
//
// Costo de construccion: O(n * m) distancias. Es alto y se reporta aparte en
// buildDistances(); un indice que gasta mucho al construirse solo compensa si
// se consulta muchas veces.
void Laesa::build(const Dataset& data, const Metric& metric) {
    data_ = &data;
    metric_ = &metric;
    const std::size_t n = data.size();
    const std::size_t m = std::min(cfg_.num_pivots, n);

    const std::size_t before = metric.calls();

    pivots_.clear();
    table_.assign(n * m, 0.0);

    std::mt19937 rng(cfg_.seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    std::vector<double> min_dist(n, std::numeric_limits<double>::infinity());
    ObjectId next = pick(rng);

    for (std::size_t j = 0; j < m; ++j) {
        pivots_.push_back(next);
        double best = -1.0;
        ObjectId best_id = next;
        for (ObjectId i = 0; i < n; ++i) {
            const double d = metric(data.at(i), data.at(next));
            table_[i * m + j] = d;
            min_dist[i] = std::min(min_dist[i], d);
            if (min_dist[i] > best) { best = min_dist[i]; best_id = i; }
        }
        next = best_id;
    }

    build_distances_ = metric.calls() - before;
}

std::vector<Neighbor> Laesa::searchKnn(const float* q, std::size_t k, Trace* trace) const {
    const std::size_t n = data_->size();
    const std::size_t m = pivots_.size();
    if (k == 0 || n == 0) return {};

    std::priority_queue<Neighbor> heap;
    double tau = std::numeric_limits<double>::infinity();

    auto consider = [&](ObjectId id, double d) {
        if (heap.size() < k)                 { heap.push({id, d}); }
        else if (d < heap.top().dist)        { heap.pop(); heap.push({id, d}); }
        if (heap.size() == k) tau = heap.top().dist;
    };

    // 1) distancias reales a los pivotes (son objetos del dataset, asi que
    //    tambien compiten como candidatos). Se marcan para no volver a
    //    evaluarlos en el paso 3, lo que ademas duplicaria entradas.
    std::vector<char> es_pivote(n, 0);
    for (ObjectId p : pivots_) es_pivote[p] = 1;

    std::vector<double> dq(m);
    for (std::size_t j = 0; j < m; ++j) {
        dq[j] = (*metric_)(q, data_->at(pivots_[j]));
        if (trace) trace->add({EventKind::ComputeDist, -1, (long long)pivots_[j], dq[j], 0, 0,
                               tau, 0, "distancia al pivote " + std::to_string(j)});
        consider(pivots_[j], dq[j]);
    }

    // 2) cota inferior de cada objeto SIN calcular su distancia real:
    //       LB(x) = max_j | d(q,p_j) - d(x,p_j) |
    //    Se deduce de la desigualdad triangular. Si LB(x) > tau, x no puede
    //    estar entre los k mejores y se descarta gratis.
    std::vector<std::pair<double, ObjectId>> lb;
    lb.reserve(n);
    for (ObjectId i = 0; i < n; ++i) {
        if (es_pivote[i]) continue;  // ya evaluado en el paso 1
        double best = 0.0;
        for (std::size_t j = 0; j < m; ++j)
            best = std::max(best, std::fabs(dq[j] - tableAt(i, j)));
        lb.emplace_back(best, i);
    }

    // 3) Se recorre en orden creciente de cota: los mas prometedores primero,
    //    para que tau se cierre cuanto antes y se descarte mas.
    //    Nota: ordenar cuesta O(n log n) de CPU pero CERO distancias. Es un
    //    ejemplo claro de por que tiempo y numero de distancias no miden lo
    //    mismo, y conviene comentarlo en el informe.
    std::sort(lb.begin(), lb.end());

    std::size_t descartados = 0;
    for (const auto& [bound, id] : lb) {
        if (heap.size() == k && bound > tau) {
            // Como la lista esta ordenada, todo lo que sigue tambien se descarta.
            descartados += 1;
            if (trace) trace->add({EventKind::Prune, -1, (long long)id, 0, bound, 0, tau, 0,
                                   "cota inferior > tau: descartado sin calcular distancia"});
            continue;
        }
        const double d = (*metric_)(q, data_->at(id));
        if (trace) trace->add({EventKind::ComputeDist, -1, (long long)id, d, bound, 0, tau, 0,
                               "candidato no descartado por la cota"});
        consider(id, d);
    }
    (void)descartados;

    std::vector<Neighbor> out;
    out.reserve(heap.size());
    while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<Neighbor> Laesa::searchRange(const float* q, double radius, Trace* trace) const {
    const std::size_t n = data_->size();
    const std::size_t m = pivots_.size();
    std::vector<Neighbor> out;

    std::vector<char> es_pivote(n, 0);
    for (ObjectId p : pivots_) es_pivote[p] = 1;

    std::vector<double> dq(m);
    for (std::size_t j = 0; j < m; ++j) {
        dq[j] = (*metric_)(q, data_->at(pivots_[j]));
        if (dq[j] <= radius) out.push_back({pivots_[j], dq[j]});
    }

    for (ObjectId i = 0; i < n; ++i) {
        if (es_pivote[i]) continue;
        double bound = 0.0;
        for (std::size_t j = 0; j < m; ++j)
            bound = std::max(bound, std::fabs(dq[j] - tableAt(i, j)));
        if (bound > radius) {
            if (trace) trace->add({EventKind::Prune, -1, (long long)i, 0, bound, 0, radius, 0,
                                   "cota inferior > radio: descartado"});
            continue;
        }
        const double d = (*metric_)(q, data_->at(i));
        if (trace) trace->add({EventKind::ComputeDist, -1, (long long)i, d, bound, 0, radius, 0,
                               "candidato no descartado"});
        if (d <= radius) out.push_back({i, d});
    }

    std::sort(out.begin(), out.end());
    return out;
}

std::size_t Laesa::memoryBytes() const {
    return table_.capacity() * sizeof(double) + pivots_.capacity() * sizeof(ObjectId);
}

}  // namespace metriclab
