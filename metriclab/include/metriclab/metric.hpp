// metriclab/metric.hpp
//
// Funcion de distancia instrumentada.
//
// DECISION DE DISENO CLAVE:
// El contador de distancias vive AQUI, dentro del objeto metrica, y no dentro
// de cada indice. Asi ningun indice puede "olvidarse" de contar y las cifras
// de todos los metodos son comparables por construccion. Si el contador
// estuviera en cada indice, un bug de conteo en uno solo invalidaria toda la
// comparacion experimental sin que nos demos cuenta.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#ifndef METRICLAB_METRIC_HPP
#define METRICLAB_METRIC_HPP

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <string>

namespace metriclab {

// Clase base de toda funcion de distancia.
//
// Invariante que asumen los indices: d debe ser una METRICA, es decir cumplir
//   1) d(x,y) >= 0           (no negatividad)
//   2) d(x,y) == 0 <=> x==y  (identidad)
//   3) d(x,y) == d(y,x)      (simetria)
//   4) d(x,z) <= d(x,y)+d(y,z)  (DESIGUALDAD TRIANGULAR)
//
// La propiedad (4) es la unica que justifica las podas. Si se usa una funcion
// que no la cumple (por ejemplo 1 - coseno), los indices siguen "funcionando"
// pero devuelven resultados incorrectos sin avisar. Por eso existe
// tests/triangle_check: validamos empiricamente la metrica antes de confiar
// en ella.
class Metric {
public:
    explicit Metric(std::size_t dim) : dim_(dim) {}
    virtual ~Metric() = default;

    // Unico punto de entrada. Cuenta y delega.
    double operator()(const float* a, const float* b) const {
        ++calls_;
        return compute(a, b);
    }

    std::size_t dim() const { return dim_; }

    // --- instrumentacion ---
    std::size_t calls() const { return calls_; }
    void reset() const { calls_ = 0; }

    virtual std::string name() const = 0;

protected:
    virtual double compute(const float* a, const float* b) const = 0;

    std::size_t dim_;

private:
    // mutable: contar no altera el significado logico del objeto, y permite
    // que las busquedas sean const.
    mutable std::size_t calls_ = 0;
};

// Distancia euclidiana (L2). Metrica valida.
class EuclideanMetric : public Metric {
public:
    explicit EuclideanMetric(std::size_t dim) : Metric(dim) {}
    std::string name() const override { return "euclidean"; }

protected:
    double compute(const float* a, const float* b) const override {
        double acc = 0.0;
        for (std::size_t i = 0; i < dim_; ++i) {
            const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            acc += diff * diff;
        }
        return std::sqrt(acc);
    }
};

// Distancia Manhattan (L1). Metrica valida.
class ManhattanMetric : public Metric {
public:
    explicit ManhattanMetric(std::size_t dim) : Metric(dim) {}
    std::string name() const override { return "manhattan"; }

protected:
    double compute(const float* a, const float* b) const override {
        double acc = 0.0;
        for (std::size_t i = 0; i < dim_; ++i)
            acc += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        return acc;
    }
};

// Distancia angular: arccos de la similitud coseno.
//
// OJO: esta es la forma CORRECTA de usar el coseno en un indice metrico.
// La forma habitual (1 - coseno) NO cumple la desigualdad triangular y por
// tanto invalida las podas. El angulo si la cumple.
// Asume vectores normalizados a norma 1 (ver Dataset::normalizeRows).
class AngularMetric : public Metric {
public:
    explicit AngularMetric(std::size_t dim) : Metric(dim) {}
    std::string name() const override { return "angular"; }

protected:
    double compute(const float* a, const float* b) const override {
        double dot = 0.0;
        for (std::size_t i = 0; i < dim_; ++i)
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        dot = std::clamp(dot, -1.0, 1.0);  // protege contra error de redondeo
        return std::acos(dot);
    }
};

}  // namespace metriclab

#endif  // METRICLAB_METRIC_HPP
