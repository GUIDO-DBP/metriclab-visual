// metriclab/apps/search_cli.cpp
//
// Busqueda sobre vectores REALES extraidos de imagenes.
//
// Lee los binarios que produce demo/extract_features.py, construye todos los
// indices registrados, resuelve cada consulta con cada uno y escribe:
//
//   metricas.csv   una fila por indice: distancias, poda, tiempo, memoria,
//                  costo de construccion y si coincide con el oraculo
//   vecinos.csv    los k vecinos de cada consulta (los usa la clasificacion
//                  k-NN y la galeria del reporte)
//   traza.json     recorrido completo del VP-Tree para la primera consulta,
//                  listo para la visualizacion paso a paso
//
// Puente provisional con Python: archivos binarios. La version final usara
// pybind11; este formato sirve mientras tanto y permite probar el motor sin
// depender de la parte web.
//
// Uso:  search_cli train.bin query.bin carpeta_salida [k=10]
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#include "metriclab/laesa.hpp"
#include "metriclab/linear_scan.hpp"
#include "metriclab/vp_tree.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace metriclab;
using Clock = std::chrono::steady_clock;

static Dataset loadBin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("no se puede abrir " + path);
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, "MLV1", 4) != 0)
        throw std::runtime_error(path + ": formato desconocido (se esperaba MLV1)");
    std::uint32_t n = 0, dim = 0;
    in.read(reinterpret_cast<char*>(&n), 4);
    in.read(reinterpret_cast<char*>(&dim), 4);
    Dataset ds(n, dim);
    if (n) in.read(reinterpret_cast<char*>(ds.at(0)), std::streamsize(n) * dim * sizeof(float));
    if (!in) throw std::runtime_error(path + ": archivo truncado");
    return ds;
}

static bool sameDistances(const std::vector<Neighbor>& a, const std::vector<Neighbor>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i].dist - b[i].dist) > 1e-6) return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "uso: " << argv[0] << " train.bin query.bin carpeta_salida [k=10]\n";
        return 2;
    }
    const std::string out = argv[3];
    const std::size_t k = argc > 4 ? std::stoul(argv[4]) : 10;

    const Dataset train = loadBin(argv[1]);
    const Dataset query = loadBin(argv[2]);
    if (train.dim() != query.dim()) {
        std::cerr << "dimensiones distintas: " << train.dim() << " vs " << query.dim() << "\n";
        return 2;
    }

    EuclideanMetric metric(train.dim());

    // --- indices registrados (aqui se anadirian MVP-Tree, BST, M-tree) ---
    std::vector<std::unique_ptr<MetricIndex>> indices;
    indices.push_back(std::make_unique<LinearScan>());
    for (auto p : {PivotPolicy::Random, PivotPolicy::MaxVariance,
                   PivotPolicy::MaxSpread, PivotPolicy::FarthestFirst}) {
        VpTreeConfig cfg; cfg.policy = p;
        indices.push_back(std::make_unique<VpTree>(cfg));
    }
    for (std::size_t m : {8u, 16u, 32u})
        indices.push_back(std::make_unique<Laesa>(LaesaConfig{m, 42}));

    // Respuestas del oraculo (busqueda secuencial)
    std::vector<std::vector<Neighbor>> oracle(query.size());

    std::ofstream csv(out + "/metricas.csv");
    csv << "indice,dist_por_consulta,poda_pct,us_por_consulta,dist_construccion,"
           "memoria_kb,coincide_oraculo\n";

    std::cout << "coleccion: " << train.size() << " vectores de dimension " << train.dim()
              << "   consultas: " << query.size() << "   k=" << k << "\n\n";
    std::cout << std::left << std::setw(26) << "indice" << std::right
              << std::setw(11) << "dist/cons" << std::setw(9) << "poda%"
              << std::setw(11) << "us/cons" << std::setw(12) << "dist build"
              << std::setw(10) << "oraculo" << "\n"
              << std::string(79, '-') << "\n";

    const VpTree* vp_ref = nullptr;

    for (std::size_t ii = 0; ii < indices.size(); ++ii) {
        auto& idx = indices[ii];
        metric.reset();
        idx->build(train, metric);

        std::vector<std::vector<Neighbor>> answers(query.size());
        metric.reset();
        const auto t0 = Clock::now();
        for (std::size_t q = 0; q < query.size(); ++q)
            answers[q] = idx->searchKnn(query.at(q), k);
        const auto t1 = Clock::now();

        const double per_q = double(metric.calls()) / double(query.size());
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                          double(query.size());
        const double poda = 100.0 * (1.0 - per_q / double(train.size()));

        bool ok = true;
        if (ii == 0) {
            oracle = answers;
        } else {
            for (std::size_t q = 0; q < query.size(); ++q)
                if (!sameDistances(oracle[q], answers[q])) { ok = false; break; }
        }

        csv << '"' << idx->name() << "\"," << per_q << ',' << poda << ',' << us << ','
            << idx->buildDistances() << ',' << idx->memoryBytes() / 1024.0 << ','
            << (ok ? "si" : "NO") << '\n';

        std::cout << std::left << std::setw(26) << idx->name() << std::right << std::fixed
                  << std::setw(11) << std::setprecision(1) << per_q
                  << std::setw(9) << std::setprecision(1) << poda
                  << std::setw(11) << std::setprecision(1) << us
                  << std::setw(12) << idx->buildDistances()
                  << std::setw(10) << (ok ? "si" : "NO") << "\n";

        // Vecinos y traza: del VP-Tree con max-varianza (la politica principal)
        if (idx->name() == "VP-Tree[max-varianza]") {
            vp_ref = dynamic_cast<const VpTree*>(idx.get());
            std::ofstream nb(out + "/vecinos.csv");
            nb << "consulta,rango,id_coleccion,distancia\n";
            for (std::size_t q = 0; q < query.size(); ++q)
                for (std::size_t r = 0; r < answers[q].size(); ++r)
                    nb << q << ',' << r << ',' << answers[q][r].id << ','
                       << std::setprecision(6) << answers[q][r].dist << '\n';

            // Para la demostracion se guarda la traza de la consulta con mas
            // podas: es la que mejor ilustra como el arbol descarta regiones.
            std::size_t best_q = 0, best_prunes = 0;
            for (std::size_t q = 0; q < query.size(); ++q) {
                Trace t;
                idx->searchKnn(query.at(q), k, &t);
                std::size_t pr = 0;
                for (const auto& e : t.events()) pr += (e.kind == EventKind::Prune);
                if (pr > best_prunes) { best_prunes = pr; best_q = q; }
            }
            Trace trace;
            idx->searchKnn(query.at(best_q), k, &trace);
            std::ofstream tj(out + "/traza.json");
            trace.writeJson(tj);
            std::ofstream(out + "/traza_consulta.txt") << best_q << "\n";
        }
    }

    if (vp_ref)
        std::cout << "\nVP-Tree[max-varianza]: " << vp_ref->nodeCount() << " nodos, altura "
                  << vp_ref->height() << "\n";
    std::cout << "escrito " << out << "/metricas.csv, vecinos.csv, traza.json\n";
    return 0;
}
