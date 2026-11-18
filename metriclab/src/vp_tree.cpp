// metriclab/src/vp_tree.cpp
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#include "metriclab/vp_tree.hpp"

#include <algorithm>
#include <limits>
#include <queue>

namespace metriclab {

std::string pivotPolicyName(PivotPolicy p) {
    switch (p) {
        case PivotPolicy::Random:        return "random";
        case PivotPolicy::MaxVariance:   return "max-varianza";
        case PivotPolicy::MaxSpread:     return "max-rango";
        case PivotPolicy::FarthestFirst: return "fft";
    }
    return "desconocida";
}

// ---------------------------------------------------------------------------
// Seleccion de pivote
// ---------------------------------------------------------------------------
//
// Devuelve la POSICION dentro de ids (no el ObjectId) del pivote elegido.
//
// Intuicion de la heuristica de Yianilos: un buen pivote es el que reparte a
// los demas objetos en un rango amplio de distancias. Si todos quedan a
// distancia parecida, la mediana no separa nada y ninguna rama se podra
// descartar. Maximizar la varianza de las distancias busca exactamente eso.
std::size_t VpTree::choosePivot(const std::vector<ObjectId>& ids) {
    const std::size_t n = ids.size();
    if (n <= 2 || cfg_.policy == PivotPolicy::Random) {
        std::uniform_int_distribution<std::size_t> pick(0, n - 1);
        return pick(rng_);
    }

    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    // FFT: el objeto mas alejado de una referencia tomada al azar.
    if (cfg_.policy == PivotPolicy::FarthestFirst) {
        const std::size_t ref = pick(rng_);
        const std::size_t m = std::min(n, cfg_.sample_test);
        std::size_t best = ref;
        double best_d = -1.0;
        for (std::size_t t = 0; t < m; ++t) {
            const std::size_t cand = pick(rng_);
            if (cand == ref) continue;
            const double d = (*metric_)(data_->at(ids[ref]), data_->at(ids[cand]));
            if (d > best_d) { best_d = d; best = cand; }
        }
        return best;
    }

    // MaxVariance / MaxSpread: se evaluan varios candidatos contra una muestra.
    const std::size_t n_cand = std::min(n, cfg_.sample_candidates);
    const std::size_t n_test = std::min(n, cfg_.sample_test);

    std::size_t best = 0;
    double best_score = -1.0;

    for (std::size_t c = 0; c < n_cand; ++c) {
        const std::size_t cand = pick(rng_);
        std::vector<double> ds;
        ds.reserve(n_test);
        for (std::size_t t = 0; t < n_test; ++t) {
            const std::size_t other = pick(rng_);
            if (other == cand) continue;
            ds.push_back((*metric_)(data_->at(ids[cand]), data_->at(ids[other])));
        }
        if (ds.empty()) continue;

        double score = 0.0;
        if (cfg_.policy == PivotPolicy::MaxVariance) {
            double mean = 0.0;
            for (double d : ds) mean += d;
            mean /= static_cast<double>(ds.size());
            for (double d : ds) score += (d - mean) * (d - mean);
            score /= static_cast<double>(ds.size());
        } else {  // MaxSpread
            const auto mm = std::minmax_element(ds.begin(), ds.end());
            score = *mm.second - *mm.first;
        }

        if (score > best_score) { best_score = score; best = cand; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Construccion
// ---------------------------------------------------------------------------

int VpTree::buildNode(std::vector<ObjectId>& ids, int depth) {
    if (ids.empty()) return -1;

    const int self = static_cast<int>(nodes_.size());
    nodes_.emplace_back();

    if (ids.size() <= cfg_.leaf_size) {
        nodes_[self].leaf = true;
        nodes_[self].pivot = ids[0];
        nodes_[self].bucket = ids;
        return self;
    }

    // 1. elegir pivote y sacarlo del conjunto
    const std::size_t pos = choosePivot(ids);
    const ObjectId pivot = ids[pos];
    std::swap(ids[pos], ids.back());
    ids.pop_back();

    // 2. distancias del pivote al resto
    std::vector<std::pair<double, ObjectId>> ds;
    ds.reserve(ids.size());
    for (ObjectId id : ids)
        ds.emplace_back((*metric_)(data_->at(pivot), data_->at(id)), id);

    // 3. mediana -> radio de corte mu
    const std::size_t mid = ds.size() / 2;
    std::nth_element(ds.begin(), ds.begin() + mid, ds.end());
    const double mu = ds[mid].first;

    std::vector<ObjectId> inside, outside;
    inside.reserve(mid + 1);
    outside.reserve(ds.size() - mid);
    for (const auto& [d, id] : ds) {
        if (d < mu) inside.push_back(id);
        else        outside.push_back(id);
    }

    // Caso degenerado: todas las distancias iguales (mu no separa nada).
    // Ocurre con datos muy concentrados, que es justo el fenomeno que el
    // proyecto estudia. Se convierte en hoja en vez de recursar infinitamente.
    if (inside.empty() || outside.empty()) {
        nodes_[self].leaf = true;
        nodes_[self].pivot = pivot;
        nodes_[self].bucket.push_back(pivot);
        for (const auto& [d, id] : ds) nodes_[self].bucket.push_back(id);
        return self;
    }

    nodes_[self].pivot = pivot;
    nodes_[self].mu = mu;
    const int l = buildNode(inside, depth + 1);
    const int r = buildNode(outside, depth + 1);
    nodes_[self].left = l;
    nodes_[self].right = r;
    return self;
}

void VpTree::build(const Dataset& data, const Metric& metric) {
    data_ = &data;
    metric_ = &metric;
    nodes_.clear();
    rng_.seed(cfg_.seed);

    const std::size_t before = metric.calls();

    std::vector<ObjectId> ids(data.size());
    for (ObjectId i = 0; i < data.size(); ++i) ids[i] = i;
    nodes_.reserve(data.size() * 2);
    root_ = buildNode(ids, 0);

    build_distances_ = metric.calls() - before;
}

// ---------------------------------------------------------------------------
// Busqueda k-NN
// ---------------------------------------------------------------------------
//
// tau = radio dinamico = distancia al k-esimo mejor encontrado hasta ahora.
// Empieza en infinito y se va cerrando; cuanto antes se cierre, mas se poda.
// De ahi que el orden de visita importe: se entra primero al lado donde cae
// la consulta, porque es donde hay mas probabilidad de encontrar vecinos
// buenos que reduzcan tau.
//
// Las dos condiciones de poda salen de la DESIGUALDAD TRIANGULAR:
//   la bola de consulta cubre el intervalo [d-tau, d+tau] de distancias al
//   pivote; el hijo interno vive en [0, mu) y el externo en [mu, inf).
//   Solo se visita un hijo si esos intervalos se cortan.

std::vector<Neighbor> VpTree::searchKnn(const float* q, std::size_t k, Trace* trace) const {
    std::priority_queue<Neighbor> heap;  // max-heap: top = el peor de los k
    double tau = std::numeric_limits<double>::infinity();
    if (k == 0 || root_ < 0) return {};

    auto consider = [&](ObjectId id, double d) {
        if (heap.size() < k) {
            heap.push({id, d});
        } else if (d < heap.top().dist) {
            heap.pop();
            heap.push({id, d});
        } else {
            return;
        }
        if (heap.size() == k) tau = heap.top().dist;
    };

    // Pila explicita con profundidad, para no depender de recursion y poder
    // emitir la traza en el orden real de visita.
    struct Frame { int node; int depth; };
    std::vector<Frame> stack{{root_, 0}};

    while (!stack.empty()) {
        const auto [ni, depth] = stack.back();
        stack.pop_back();
        if (ni < 0) continue;
        const Node& nd = nodes_[ni];

        if (trace) trace->add({EventKind::VisitNode, ni, -1, 0, 0, nd.mu, tau, depth,
                               "se entra al nodo"});

        if (nd.leaf) {
            for (ObjectId id : nd.bucket) {
                const double d = (*metric_)(q, data_->at(id));
                if (trace) trace->add({EventKind::ComputeDist, ni, (long long)id, d, 0, 0, tau,
                                       depth, "objeto en hoja"});
                const bool entra = heap.size() < k || d < heap.top().dist;
                consider(id, d);
                if (trace && entra)
                    trace->add({EventKind::Candidate, ni, (long long)id, d, 0, 0, tau, depth,
                                "entra a los k mejores"});
            }
            continue;
        }

        const double d = (*metric_)(q, data_->at(nd.pivot));
        if (trace) trace->add({EventKind::ComputeDist, ni, (long long)nd.pivot, d, 0, nd.mu, tau,
                               depth, "distancia al pivote"});
        const bool entra = heap.size() < k || d < heap.top().dist;
        consider(nd.pivot, d);
        if (trace && entra)
            trace->add({EventKind::Candidate, ni, (long long)nd.pivot, d, 0, nd.mu, tau, depth,
                        "el pivote entra a los k mejores"});

        const bool want_in  = (d - tau <  nd.mu);
        const bool want_out = (d + tau >= nd.mu);

        if (trace) {
            if (!want_in && nd.left >= 0)
                trace->add({EventKind::Prune, nd.left, -1, d, d - tau, nd.mu, tau, depth + 1,
                            "se descarta el hijo interno: d-tau >= mu"});
            if (!want_out && nd.right >= 0)
                trace->add({EventKind::Prune, nd.right, -1, d, d + tau, nd.mu, tau, depth + 1,
                            "se descarta el hijo externo: d+tau < mu"});
        }

        // Se apila primero el lado lejano para desapilar primero el cercano.
        if (d < nd.mu) {
            if (want_out && nd.right >= 0) stack.push_back({nd.right, depth + 1});
            if (want_in  && nd.left  >= 0) stack.push_back({nd.left,  depth + 1});
        } else {
            if (want_in  && nd.left  >= 0) stack.push_back({nd.left,  depth + 1});
            if (want_out && nd.right >= 0) stack.push_back({nd.right, depth + 1});
        }
    }

    std::vector<Neighbor> out;
    out.reserve(heap.size());
    while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<Neighbor> VpTree::searchRange(const float* q, double radius, Trace* trace) const {
    std::vector<Neighbor> out;
    if (root_ < 0) return out;

    struct Frame { int node; int depth; };
    std::vector<Frame> stack{{root_, 0}};

    while (!stack.empty()) {
        const auto [ni, depth] = stack.back();
        stack.pop_back();
        if (ni < 0) continue;
        const Node& nd = nodes_[ni];
        if (trace) trace->add({EventKind::VisitNode, ni, -1, 0, 0, nd.mu, radius, depth,
                               "se entra al nodo"});

        if (nd.leaf) {
            for (ObjectId id : nd.bucket) {
                const double d = (*metric_)(q, data_->at(id));
                if (trace) trace->add({EventKind::ComputeDist, ni, (long long)id, d, radius, 0,
                                       radius, depth, "objeto en hoja"});
                if (d <= radius) out.push_back({id, d});
            }
            continue;
        }

        const double d = (*metric_)(q, data_->at(nd.pivot));
        if (trace) trace->add({EventKind::ComputeDist, ni, (long long)nd.pivot, d, radius, nd.mu,
                               radius, depth, "distancia al pivote"});
        if (d <= radius) out.push_back({nd.pivot, d});

        if (d - radius <  nd.mu && nd.left  >= 0) stack.push_back({nd.left,  depth + 1});
        if (d + radius >= nd.mu && nd.right >= 0) stack.push_back({nd.right, depth + 1});
    }

    std::sort(out.begin(), out.end());
    return out;
}

// ---------------------------------------------------------------------------

std::size_t VpTree::memoryBytes() const {
    std::size_t total = nodes_.capacity() * sizeof(Node);
    for (const auto& n : nodes_) total += n.bucket.capacity() * sizeof(ObjectId);
    return total;
}

int VpTree::heightFrom(int node) const {
    if (node < 0) return 0;
    const Node& nd = nodes_[node];
    if (nd.leaf) return 1;
    return 1 + std::max(heightFrom(nd.left), heightFrom(nd.right));
}

int VpTree::height() const { return heightFrom(root_); }

void VpTree::writeStructure(std::ostream& os) const {
    os << "[";
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (i) os << ",";
        const auto& n = nodes_[i];
        os << "{\"id\":" << i << ",\"pivot\":" << n.pivot
           << ",\"radius\":" << n.mu << ",\"left\":" << n.left
           << ",\"right\":" << n.right << ",\"leaf\":" << (n.leaf ? "true" : "false") << "}";
    }
    os << "]";
}

}  // namespace metriclab
