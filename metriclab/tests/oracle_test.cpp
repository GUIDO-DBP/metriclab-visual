// metriclab/tests/oracle_test.cpp
//
// TEST ORACULO.
//
// Todo indice EXACTO debe devolver exactamente el mismo resultado que la
// busqueda secuencial. Es la prueba que convierte "mi indice es mas rapido"
// en una afirmacion defendible: sin ella, un bug en la condicion de poda
// hace que un indice visite menos nodos y parezca mejor, cuando en realidad
// esta perdiendo vecinos.
//
// Se ejecuta sobre varias configuraciones: dimensiones distintas, datos
// agrupados y uniformes, y las cuatro politicas de pivote.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#include "metriclab/laesa.hpp"
#include "metriclab/linear_scan.hpp"
#include "metriclab/synthetic.hpp"
#include "metriclab/vp_tree.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace metriclab;

namespace {

int fallos = 0;
int pruebas = 0;

void check(bool cond, const std::string& msg) {
    ++pruebas;
    if (!cond) {
        ++fallos;
        std::cout << "  [FALLO] " << msg << "\n";
    }
}

// Compara por DISTANCIA, no por id: si hay empates exactos, dos metodos
// correctos pueden devolver objetos distintos y ambos son validos.
bool mismoResultado(const std::vector<Neighbor>& a, const std::vector<Neighbor>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i].dist - b[i].dist) > 1e-9) return false;
    return true;
}

void probarConfiguracion(const std::string& etiqueta, const Dataset& ds,
                         const Metric& metric, std::size_t k) {
    LinearScan oraculo;
    oraculo.build(ds, metric);

    std::vector<std::unique_ptr<MetricIndex>> indices;
    for (auto p : {PivotPolicy::Random, PivotPolicy::MaxVariance,
                   PivotPolicy::MaxSpread, PivotPolicy::FarthestFirst}) {
        VpTreeConfig cfg;
        cfg.policy = p;
        indices.push_back(std::make_unique<VpTree>(cfg));
    }
    { VpTreeConfig cfg; cfg.policy = PivotPolicy::MaxVariance; cfg.leaf_size = 8;
      indices.push_back(std::make_unique<VpTree>(cfg)); }
    indices.push_back(std::make_unique<Laesa>(LaesaConfig{8, 42}));
    indices.push_back(std::make_unique<Laesa>(LaesaConfig{24, 42}));

    for (auto& idx : indices) idx->build(ds, metric);

    // 25 consultas tomadas del propio dataset y 25 externas.
    std::mt19937 rng(99);
    std::uniform_int_distribution<std::size_t> pick(0, ds.size() - 1);
    Dataset externas = makeUniform(25, ds.dim(), 777);

    for (int ronda = 0; ronda < 50; ++ronda) {
        const float* q = (ronda < 25) ? ds.at(pick(rng)) : externas.at(ronda - 25);

        const auto esperado = oraculo.searchKnn(q, k);
        for (auto& idx : indices) {
            const auto obtenido = idx->searchKnn(q, k);
            check(mismoResultado(esperado, obtenido),
                  etiqueta + " / " + idx->name() + " / kNN consulta " + std::to_string(ronda));
        }

        // Busqueda por rango: radio tal que caiga una fraccion de los datos.
        const double radio = esperado.empty() ? 0.0 : esperado.back().dist;
        const auto esperado_r = oraculo.searchRange(q, radio);
        for (auto& idx : indices) {
            const auto obtenido_r = idx->searchRange(q, radio);
            check(obtenido_r.size() == esperado_r.size(),
                  etiqueta + " / " + idx->name() + " / rango consulta " + std::to_string(ronda) +
                  " (esperado " + std::to_string(esperado_r.size()) +
                  ", obtenido " + std::to_string(obtenido_r.size()) + ")");
        }
    }
}

}  // namespace

int main() {
    std::cout << "=== TEST ORACULO: todo indice exacto == busqueda secuencial ===\n\n";

    struct Caso { std::string nombre; Dataset ds; };
    std::vector<Caso> casos;
    casos.push_back({"agrupado d=8   n=600",  makeClustered(600, 8,  10)});
    casos.push_back({"agrupado d=64  n=600",  makeClustered(600, 64, 10)});
    casos.push_back({"uniforme d=16  n=500",  makeUniform(500, 16)});
    casos.push_back({"uniforme d=128 n=400",  makeUniform(400, 128)});
    // Caso adversario: muchos duplicados exactos. Rompe la particion por
    // mediana y es el que descubre los errores en el caso degenerado.
    {
        Dataset dup = makeClustered(300, 4, 3, 0.0001);
        casos.push_back({"casi duplicados d=4 n=300", dup});
    }

    for (auto& caso : casos) {
        std::cout << "-- " << caso.nombre << "\n";
        EuclideanMetric metric(caso.ds.dim());
        for (std::size_t k : {1u, 5u, 20u}) probarConfiguracion(caso.nombre, caso.ds, metric, k);
    }

    // Validacion de la desigualdad triangular para cada metrica disponible.
    std::cout << "\n-- validacion de metricas (desigualdad triangular)\n";
    Dataset ds = makeClustered(400, 32, 6);
    {
        EuclideanMetric m(32);
        const auto v = triangleViolations(ds, m);
        std::cout << "   euclidiana : " << v << " violaciones\n";
        check(v == 0, "la euclidiana debe cumplir la desigualdad triangular");
    }
    {
        ManhattanMetric m(32);
        const auto v = triangleViolations(ds, m);
        std::cout << "   manhattan  : " << v << " violaciones\n";
        check(v == 0, "manhattan debe cumplir la desigualdad triangular");
    }
    {
        Dataset norm = ds;
        norm.normalizeRows();
        AngularMetric m(32);
        const auto v = triangleViolations(norm, m);
        std::cout << "   angular    : " << v << " violaciones\n";
        check(v == 0, "la distancia angular debe cumplir la desigualdad triangular");
    }

    std::cout << "\n==================================================\n";
    std::cout << "pruebas: " << pruebas << "   fallos: " << fallos << "\n";
    if (fallos == 0) std::cout << "RESULTADO: todos los indices coinciden con el oraculo.\n";
    else             std::cout << "RESULTADO: HAY FALLOS. No usar estas cifras en el informe.\n";
    return fallos == 0 ? 0 : 1;
}
