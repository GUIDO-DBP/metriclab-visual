// metriclab/dataset.hpp
//
// Coleccion de vectores en memoria contigua (n x dim, row-major).
//
// Los indices nunca copian vectores: guardan identificadores enteros y piden
// el puntero al Dataset. Asi el consumo de memoria que reporta cada indice
// (memoryBytes) mide SOLO la estructura de indexacion, no los datos, que es
// lo que queremos comparar.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#ifndef METRICLAB_DATASET_HPP
#define METRICLAB_DATASET_HPP

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace metriclab {

using ObjectId = std::size_t;

class Dataset {
public:
    Dataset() = default;
    Dataset(std::size_t n, std::size_t dim) : dim_(dim), data_(n * dim, 0.0f), labels_(n) {}

    std::size_t size() const { return dim_ ? data_.size() / dim_ : 0; }
    std::size_t dim() const { return dim_; }

    const float* at(ObjectId id) const { return data_.data() + id * dim_; }
    float* at(ObjectId id) { return data_.data() + id * dim_; }

    // Etiqueta de clase (categoria de Caltech-101, por ejemplo).
    // Se usa solo para medir Precision@k; los indices nunca la ven.
    const std::string& label(ObjectId id) const { return labels_[id]; }
    void setLabel(ObjectId id, std::string value) { labels_[id] = std::move(value); }

    void push(const std::vector<float>& v, std::string lab = {}) {
        if (dim_ == 0) dim_ = v.size();
        if (v.size() != dim_) throw std::invalid_argument("Dataset::push: dimension incorrecta");
        data_.insert(data_.end(), v.begin(), v.end());
        labels_.push_back(std::move(lab));
    }

    // Normaliza cada fila a norma 1. Requisito para usar AngularMetric.
    void normalizeRows() {
        for (std::size_t i = 0; i < size(); ++i) {
            float* row = at(i);
            double norm = 0.0;
            for (std::size_t j = 0; j < dim_; ++j) norm += double(row[j]) * double(row[j]);
            norm = std::sqrt(norm);
            if (norm <= 1e-12) continue;
            for (std::size_t j = 0; j < dim_; ++j) row[j] = static_cast<float>(row[j] / norm);
        }
    }

    // Devuelve una vista con solo las primeras `d` dimensiones.
    // Es el mecanismo del experimento de dimensionalidad: mismos objetos,
    // distinta dimension, sin volver a extraer caracteristicas.
    Dataset project(std::size_t d) const {
        if (d > dim_) throw std::invalid_argument("Dataset::project: d > dim");
        Dataset out(size(), d);
        for (std::size_t i = 0; i < size(); ++i) {
            const float* src = at(i);
            float* dst = out.at(i);
            for (std::size_t j = 0; j < d; ++j) dst[j] = src[j];
            out.setLabel(i, labels_[i]);
        }
        return out;
    }

private:
    std::size_t dim_ = 0;
    std::vector<float> data_;
    std::vector<std::string> labels_;
};

}  // namespace metriclab

#endif  // METRICLAB_DATASET_HPP
