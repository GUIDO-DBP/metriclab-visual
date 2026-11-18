// metriclab/bench/bench_main.cpp
//
// ARNES DE BENCHMARK.
//
// Recorre cualquier indice que implemente MetricIndex. Cuando se anada
// MVP-Tree, BST o M-tree, basta con registrarlo en la lista: no hay que
// tocar nada mas de este archivo.
//
// Metrica principal: DISTANCIAS CALCULADAS por consulta. El tiempo depende
// de la maquina, la cache y el compilador; el numero de distancias no, y es
// la unidad estandar en la literatura de espacios metricos. El tiempo se
// reporta como secundario.
//
// Salida en CSV para graficar directamente desde Python.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#include "metriclab/laesa.hpp"
#include "metriclab/linear_scan.hpp"
#include "metriclab/synthetic.hpp"
#include "metriclab/vp_tree.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace metriclab;
using Clock = std::chrono::steady_clock;

struct Fila {
    std::string dataset;
    std::size_t n, dim;
    double rho;             // dimensionalidad intrinseca
    std::string indice;
    std::size_t k;
    double dist_consulta;   // media de distancias por consulta
    double poda_pct;        // % de objetos evitados frente al secuencial
    double us_consulta;     // microsegundos por consulta
    std::size_t dist_build;
    std::size_t mem_kb;
};

static std::vector<std::unique_ptr<MetricIndex>> crearIndices() {
    std::vector<std::unique_ptr<MetricIndex>> v;
    v.push_back(std::make_unique<LinearScan>());
    for (auto p : {PivotPolicy::Random, PivotPolicy::MaxVariance,
                   PivotPolicy::MaxSpread, PivotPolicy::FarthestFirst}) {
        VpTreeConfig cfg; cfg.policy = p;
        v.push_back(std::make_unique<VpTree>(cfg));
    }
    v.push_back(std::make_unique<Laesa>(LaesaConfig{16, 42}));
    v.push_back(std::make_unique<Laesa>(LaesaConfig{48, 42}));
    // Aqui se registran las ampliaciones futuras (MVP-Tree, BST, M-tree).
    return v;
}

static void correr(const std::string& nombre, const Dataset& ds, const Dataset& consultas,
                   std::size_t k, std::vector<Fila>& salida) {
    EuclideanMetric metric(ds.dim());
    const auto stats = distanceStats(ds, metric);

    for (auto& idx : crearIndices()) {
        metric.reset();
        idx->build(ds, metric);
        const std::size_t d_build = idx->buildDistances();

        metric.reset();
        const auto t0 = Clock::now();
        for (std::size_t i = 0; i < consultas.size(); ++i)
            idx->searchKnn(consultas.at(i), k);
        const auto t1 = Clock::now();

        const double por_consulta =
            double(metric.calls()) / double(consultas.size());
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                          double(consultas.size());

        salida.push_back({nombre, ds.size(), ds.dim(), stats.intrinsic_dim, idx->name(), k,
                          por_consulta,
                          100.0 * (1.0 - por_consulta / double(ds.size())),
                          us, d_build, idx->memoryBytes() / 1024});
    }
}

int main() {
    std::vector<Fila> filas;

    std::cout << "Ejecutando banco de pruebas...\n\n";

    // Experimento 1: efecto de la DIMENSION con todo lo demas fijo.
    // Consultas de la misma distribucion que los datos, pero fuera del indice.
    for (std::size_t d : {4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
        auto [ds, qs] = makeClusteredWithQueries(5000, 100, d, 20);
        correr("agrupado", ds, qs, 10, filas);
    }

    // Experimento 2: datos uniformes, el caso adverso (sin estructura que
    // explotar). Aqui las consultas uniformes SI son la distribucion correcta.
    for (std::size_t d : {4u, 16u, 64u, 256u})
        correr("uniforme", makeUniform(5000, d), makeUniform(100, d, 4242), 10, filas);

    // Experimento 3: efecto del TAMANO con dimension fija.
    for (std::size_t n : {1000u, 2500u, 5000u, 10000u}) {
        auto [ds, qs] = makeClusteredWithQueries(n, 100, 32, 20);
        correr("escala", ds, qs, 10, filas);
    }

    // --- CSV ---
    std::ofstream csv("resultados.csv");
    csv << "dataset,n,dim,rho,indice,k,dist_por_consulta,poda_pct,us_por_consulta,"
           "dist_construccion,memoria_kb\n";
    for (const auto& f : filas)
        csv << f.dataset << ',' << f.n << ',' << f.dim << ',' << f.rho << ','
            << '"' << f.indice << '"' << ',' << f.k << ',' << f.dist_consulta << ','
            << f.poda_pct << ',' << f.us_consulta << ',' << f.dist_build << ','
            << f.mem_kb << '\n';

    // --- Resumen legible: el experimento de dimensionalidad ---
    std::cout << "EXPERIMENTO 1 - datos agrupados, n=5000, k=10\n";
    std::cout << "El % de poda es cuanto se ahorra frente a revisar los 5000 objetos.\n\n";
    std::cout << std::left << std::setw(6) << "dim" << std::setw(10) << "rho"
              << std::setw(26) << "indice" << std::right << std::setw(12) << "dist/cons"
              << std::setw(10) << "poda%" << "\n";
    std::cout << std::string(64, '-') << "\n";
    for (const auto& f : filas) {
        if (f.dataset != "agrupado") continue;
        std::cout << std::left << std::setw(6) << f.dim
                  << std::setw(10) << std::fixed << std::setprecision(2) << f.rho
                  << std::setw(26) << f.indice
                  << std::right << std::setw(12) << std::setprecision(0) << f.dist_consulta
                  << std::setw(10) << std::setprecision(1) << f.poda_pct << "\n";
    }

    std::cout << "\nEscrito resultados.csv (" << filas.size() << " filas)\n";
    return 0;
}
