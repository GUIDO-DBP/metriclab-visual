// metriclab/synthetic.hpp
//
// Generador de datos sinteticos para pruebas y para el experimento de
// dimensionalidad, mientras no esten listos los descriptores de Caltech-101.
//
// Se generan grupos gaussianos porque asi se puede CONTROLAR la dimension
// intrinseca: con grupos bien separados hay estructura que explotar, y con
// datos uniformes en dimension alta las distancias se concentran y ningun
// indice puede podar. Poder fabricar ambos extremos permite validar el motor
// antes de depender del dataset real.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#ifndef METRICLAB_SYNTHETIC_HPP
#define METRICLAB_SYNTHETIC_HPP

#include "metriclab/dataset.hpp"
#include "metriclab/metric.hpp"

#include <cmath>
#include <random>
#include <string>
#include <utility>

namespace metriclab {

// n objetos en dim dimensiones, repartidos en `clusters` grupos gaussianos.
inline Dataset makeClustered(std::size_t n, std::size_t dim, std::size_t clusters,
                             double spread = 0.12, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, static_cast<float>(spread));

    std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
    for (auto& c : centers)
        for (auto& v : c) v = uni(rng);

    Dataset ds(n, dim);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = i % clusters;
        float* row = ds.at(i);
        for (std::size_t j = 0; j < dim; ++j) row[j] = centers[c][j] + gauss(rng);
        ds.setLabel(i, "c" + std::to_string(c));
    }
    return ds;
}

// Dataset y CONSULTAS generados desde los MISMOS centros.
//
// Importante metodologicamente: si las consultas salen de otra distribucion
// que los datos, la consulta cae lejos de todo, el radio dinamico tau nunca
// se cierra y la poda parece peor de lo que es. Las consultas deben venir de
// la misma distribucion que la coleccion, pero NO estar dentro de ella (una
// consulta que esta en el indice tiene un vecino a distancia 0 y sesga el
// resultado). Por eso se generan aparte, desde los mismos centros.
inline std::pair<Dataset, Dataset> makeClusteredWithQueries(
        std::size_t n, std::size_t n_query, std::size_t dim, std::size_t clusters,
        double spread = 0.12, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, static_cast<float>(spread));

    std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
    for (auto& c : centers)
        for (auto& v : c) v = uni(rng);

    auto emit = [&](std::size_t count) {
        Dataset ds(count, dim);
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t c = i % clusters;
            float* row = ds.at(i);
            for (std::size_t j = 0; j < dim; ++j) row[j] = centers[c][j] + gauss(rng);
            ds.setLabel(i, "c" + std::to_string(c));
        }
        return ds;
    };
    Dataset base = emit(n);
    Dataset queries = emit(n_query);
    return {std::move(base), std::move(queries)};
}

// Datos uniformes: el caso dificil. En dimension alta las distancias se
// concentran y los indices metricos degeneran hacia la busqueda secuencial.
inline Dataset makeUniform(std::size_t n, std::size_t dim, unsigned seed = 11) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    Dataset ds(n, dim);
    for (std::size_t i = 0; i < n; ++i) {
        float* row = ds.at(i);
        for (std::size_t j = 0; j < dim; ++j) row[j] = uni(rng);
        ds.setLabel(i, "u");
    }
    return ds;
}

// ---------------------------------------------------------------------------
// Dimensionalidad intrinseca (Chavez & Navarro):
//
//      rho = mu^2 / (2 * sigma^2)
//
// sobre el histograma de distancias entre pares. Un rho ALTO significa que
// todas las distancias se parecen entre si, las cotas inferiores dejan de
// discriminar y la poda se vuelve inutil. Es la magnitud que explica POR QUE
// un indice funciona en unos datos y no en otros, y es el hilo conductor del
// analisis experimental del proyecto.
// ---------------------------------------------------------------------------
struct DistanceStats {
    double mean = 0.0;
    double variance = 0.0;
    double intrinsic_dim = 0.0;
    std::size_t pairs = 0;
};

inline DistanceStats distanceStats(const Dataset& ds, const Metric& metric,
                                   std::size_t sample_pairs = 20000, unsigned seed = 3) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, ds.size() - 1);

    double sum = 0.0, sum2 = 0.0;
    std::size_t count = 0;
    for (std::size_t t = 0; t < sample_pairs; ++t) {
        const std::size_t a = pick(rng), b = pick(rng);
        if (a == b) continue;
        const double d = metric(ds.at(a), ds.at(b));
        sum += d;
        sum2 += d * d;
        ++count;
    }
    DistanceStats st;
    if (count == 0) return st;
    st.pairs = count;
    st.mean = sum / static_cast<double>(count);
    st.variance = sum2 / static_cast<double>(count) - st.mean * st.mean;
    if (st.variance > 1e-15) st.intrinsic_dim = (st.mean * st.mean) / (2.0 * st.variance);
    return st;
}

// ---------------------------------------------------------------------------
// Validacion empirica de la desigualdad triangular.
//
// Media pagina de informe que demuestra criterio: antes de confiar en las
// podas, se comprueba sobre trios al azar que d(x,z) <= d(x,y) + d(y,z).
// Si una funcion falla aqui, los indices devolveran resultados incorrectos
// sin dar ningun error.
// ---------------------------------------------------------------------------
inline std::size_t triangleViolations(const Dataset& ds, const Metric& metric,
                                      std::size_t trials = 5000, unsigned seed = 5) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, ds.size() - 1);
    std::size_t bad = 0;
    for (std::size_t t = 0; t < trials; ++t) {
        const std::size_t x = pick(rng), y = pick(rng), z = pick(rng);
        const double dxz = metric(ds.at(x), ds.at(z));
        const double dxy = metric(ds.at(x), ds.at(y));
        const double dyz = metric(ds.at(y), ds.at(z));
        if (dxz > dxy + dyz + 1e-9) ++bad;
    }
    return bad;
}

}  // namespace metriclab

#endif  // METRICLAB_SYNTHETIC_HPP
