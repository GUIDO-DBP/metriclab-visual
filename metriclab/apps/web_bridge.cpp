// Minimal C ABI: Python owns lifetime and serializes access to this engine.
// Existing MetricIndex implementations remain the only search implementations.
#include "metriclab/laesa.hpp"
#include "metriclab/linear_scan.hpp"
#include "metriclab/vp_tree.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
using namespace metriclab;
using Clock = std::chrono::steady_clock;
struct Engine {
    Dataset data;
    EuclideanMetric metric;
    std::vector<std::unique_ptr<MetricIndex>> indices;
    std::vector<double> build_ms;
    Engine(const float* values, std::size_t n, std::size_t d): data(n,d), metric(d) {
        std::memcpy(data.at(0), values, n*d*sizeof(float));
        indices.push_back(std::make_unique<LinearScan>());
        indices.push_back(std::make_unique<VpTree>());
        indices.push_back(std::make_unique<Laesa>(LaesaConfig{16,42}));
        for (auto& idx: indices) {
            auto t = Clock::now(); idx->build(data, metric);
            build_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-t).count());
        }
    }
};
thread_local std::string output, error;
API const char* ml_error() { return error.c_str(); }
API void* ml_create(const float* values, unsigned n, unsigned d) {
    try {
        if (!values || !n || d!=144) throw std::runtime_error("Expected nonempty 144D dataset");
        return new Engine(values,n,d);
    } catch (const std::exception& e) { error=e.what(); return nullptr; }
}
API void ml_destroy(void* ptr) { delete static_cast<Engine*>(ptr); }
API const char* ml_structure(void* ptr) {
    std::ostringstream os; static_cast<VpTree*>(static_cast<Engine*>(ptr)->indices[1].get())->writeStructure(os);
    output=os.str(); return output.c_str();
}
API const char* ml_search(void* ptr, const float* q, unsigned k, int selected) {
    try {
        auto& e=*static_cast<Engine*>(ptr);
        if (!k || k>e.data.size() || selected < -1 || selected > 2)
            throw std::runtime_error("Invalid k or algorithm");
        // Oracle verification is outside the measured query.
        auto oracle=e.indices[0]->searchKnn(q,k);
        std::ostringstream os; os << std::setprecision(15) << "{\"runs\":[";
        bool first=true;
        for (int i=0;i<3;++i) {
            if (selected!=-1 && selected!=i) continue;
            e.metric.reset(); auto start=Clock::now();
            auto neighbors=e.indices[i]->searchKnn(q,k);
            double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            auto calls=e.metric.calls(); bool exact=neighbors.size()==oracle.size();
            for (std::size_t j=0; exact && j<neighbors.size(); ++j)
                exact=std::abs(neighbors[j].dist-oracle[j].dist)<1e-9;
            if (!exact) throw std::runtime_error("Index differs from sequential oracle");
            if (!first) os << ","; first=false;
            os << "{\"algorithm\":" << i << ",\"query_ms\":" << ms
               << ",\"distances\":" << calls << ",\"avoided_pct\":" << 100.0*(1.0-double(calls)/e.data.size())
               << ",\"build_ms\":" << e.build_ms[i] << ",\"build_distances\":" << e.indices[i]->buildDistances()
               << ",\"memory_bytes\":" << e.indices[i]->memoryBytes() << ",\"oracle_match\":true,\"neighbors\":[";
            for (std::size_t j=0;j<neighbors.size();++j) {
                if (j) os << ",";
                os << "{\"id\":" << neighbors[j].id << ",\"distance\":" << neighbors[j].dist << "}";
            }
            os << "]}";
        }
        os << "],\"trace\":";
        if (selected==1 || selected==-1) {
            Trace trace; e.indices[1]->searchKnn(q,k,&trace); trace.writeJson(os);
        } else os << "[]";
        os << "}"; output=os.str(); return output.c_str();
    } catch (const std::exception& ex) { error=ex.what(); return nullptr; }
}
