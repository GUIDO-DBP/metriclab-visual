// metriclab/src/trace.cpp
//
// Serializacion de la traza a JSON.
//
// Este archivo es el puente con la interfaz web: el motor C++ reporta lo que
// REALMENTE ocurrio y el navegador solo lo dibuja. La visualizacion nunca
// reimplementa la logica de poda, asi que lo que se ve en pantalla no puede
// discrepar del algoritmo.
//
// MetricLab Visual - Estructuras de Datos Avanzadas (SIS220)

#include "metriclab/index.hpp"

#include <limits>
#include <ostream>

namespace metriclab {
namespace {

const char* kindName(EventKind k) {
    switch (k) {
        case EventKind::VisitNode:   return "visit";
        case EventKind::ComputeDist: return "dist";
        case EventKind::Prune:       return "prune";
        case EventKind::Candidate:   return "candidate";
        case EventKind::Descend:     return "descend";
    }
    return "unknown";
}

void writeEscaped(std::ostream& os, const std::string& s) {
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n";  break;
            default:   os << c;
        }
    }
}

// tau puede ser infinito al inicio de una busqueda; JSON no admite Infinity.
void writeNumber(std::ostream& os, double v) {
    if (v == std::numeric_limits<double>::infinity())       os << "null";
    else if (v == -std::numeric_limits<double>::infinity()) os << "null";
    else if (v != v)                                        os << "null";  // NaN
    else                                                    os << v;
}

}  // namespace

void Trace::writeJson(std::ostream& os) const {
    os << "[\n";
    for (std::size_t i = 0; i < events_.size(); ++i) {
        const TraceEvent& e = events_[i];
        os << "  {\"step\":" << i
           << ",\"kind\":\"" << kindName(e.kind) << "\""
           << ",\"node\":" << e.node
           << ",\"object\":" << e.object
           << ",\"dist\":";   writeNumber(os, e.dist);
        os << ",\"bound\":";  writeNumber(os, e.bound);
        os << ",\"radius\":"; writeNumber(os, e.radius);
        os << ",\"tau\":";    writeNumber(os, e.tau);
        os << ",\"depth\":" << e.depth
           << ",\"reason\":\"";
        writeEscaped(os, e.reason);
        os << "\"}";
        if (i + 1 < events_.size()) os << ",";
        os << "\n";
    }
    os << "]\n";
}

}  // namespace metriclab
