// QAPG-R: cyclic block descent for the exact one-slot quadratic objective.
// Independent orthogonal links; all queue/service arguments use Mbit.
// Build without -ffast-math: reproducibility and monotonicity are audited.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
struct Problem {
    int n, m;
    const double *q, *h, *g;
    double v, arrival, kd, ke, tau, noise, k, pmax, lmax, dmax, edge_weight;
};

double positive_root(double rhs, double cubic_weight) {
    if (rhs <= 0.0) return 0.0;
    // 3*c*x^2 + x = rhs, cancellation-free positive root.
    return 2.0 * rhs / (1.0 + std::sqrt(1.0 + 12.0*cubic_weight*rhs));
}

double local_given_upload(const Problem &p, int i, double upload) {
    double available = std::max(0.0, p.q[i] - upload);
    return std::min({p.lmax, available,
        positive_root(p.q[i] + p.arrival - upload, p.v*p.kd)});
}

struct Candidate { int server; double local, upload, cost; };

double device_cost(const Problem &p, int i, int server, double local,
                   double upload, double others) {
    const double residual = p.q[i] + p.arrival - local - upload;
    double cost = 0.5*residual*residual + p.v*p.kd*local*local*local;
    if (server >= 0) {
        const double radio = p.tau*p.noise/p.g[i*p.m + server];
        cost += p.v*radio*std::expm1(p.k*upload)
              + p.edge_weight*(others*upload + 0.5*upload*upload);
    }
    return cost;
}

Candidate minimize_server(const Problem &p, int i, int server, double others) {
    if (server < 0) {
        const double local = local_given_upload(p, i, 0.0);
        return {-1, local, 0.0, device_cost(p, i, -1, local, 0.0, 0.0)};
    }
    const double gain = p.g[i*p.m + server];
    if (gain <= 0.0 || p.pmax <= 0.0 || p.q[i] <= 0.0)
        return {server, 0.0, 0.0, std::numeric_limits<double>::infinity()};
    const double upper = std::min(p.q[i], std::log1p(gain*p.pmax/p.noise)/p.k);
    const double tx_derivative_scale = p.v*p.tau*p.noise/gain*p.k;
    const auto derivative = [&](double upload) {
        const double local = local_given_upload(p, i, upload);
        const double available = std::max(0.0, p.q[i] - upload);
        const double radio = tx_derivative_scale*std::exp(p.k*upload);
        // At l=q-u the available-work multiplier changes the envelope
        // derivative. Both expressions agree at the transition point.
        if (available <= p.lmax &&
            available <= positive_root(p.q[i]+p.arrival-upload, p.v*p.kd))
            return p.edge_weight*(others + upload) + radio - 3.0*p.v*p.kd*local*local;
        return -(p.q[i]+p.arrival) + local + upload + p.edge_weight*(others+upload) + radio;
    };
    double upload;
    if (derivative(0.0) >= 0.0) upload = 0.0;
    else if (derivative(upper) <= 0.0) upload = upper;
    else {
        double lower = 0.0, higher = upper;
        for (int iteration=0; iteration<40 && higher-lower>1e-10; ++iteration) {
            const double middle = 0.5*(lower+higher);
            if (derivative(middle) > 0.0) higher = middle;
            else lower = middle;
        }
        upload = 0.5*(lower+higher);
    }
    const double local = local_given_upload(p, i, upload);
    return {server, local, upload, device_cost(p, i, server, local, upload, others)};
}

std::vector<double> server_loads(const Problem &p, const double *u, const int *a) {
    std::vector<double> loads(p.m, 0.0);
    for (int i=0; i<p.n; ++i) if (a[i]>=0) loads[a[i]] += u[i];
    return loads;
}

long double objective(const Problem &p, const double *l, const double *u,
                      const double *d, const int *a) {
    const auto loads = server_loads(p, u, a);
    long double value = 0.0L;
    for (int i=0; i<p.n; ++i) {
        const long double residual = (long double)p.q[i]+p.arrival-l[i]-u[i];
        value += 0.5L*residual*residual + (long double)p.v*p.kd*l[i]*l[i]*l[i];
        if (a[i]>=0)
            value += (long double)p.v*p.tau*p.noise/p.g[i*p.m+a[i]]*std::expm1(p.k*u[i]);
    }
    for (int j=0; j<p.m; ++j) {
        const long double residual = (long double)p.h[j]-d[j]+loads[j];
        value += 0.5L*p.edge_weight*residual*residual + (long double)p.v*p.ke*d[j]*d[j]*d[j];
    }
    return value;
}

void update_edges(const Problem &p, const std::vector<double> &loads, double *d) {
    for (int j=0; j<p.m; ++j)
        d[j] = std::min({p.h[j], p.dmax,
            positive_root(p.h[j]+loads[j], p.v*p.ke/p.edge_weight)});
}
}

extern "C" int qapg_optimize(int n, int m, const double *q, const double *h,
        const double *g, const double *parameters, int max_sweeps, double tolerance,
        double *l, double *u, double *d, int *a, double *statistics) {
    try {
        const Problem p{n,m,q,h,g,parameters[0],parameters[1],parameters[2],parameters[3],
            parameters[4],parameters[5],parameters[6],parameters[7],parameters[8],parameters[9],parameters[10]};
        for (int i=0; i<n; ++i) { a[i]=-1; u[i]=0.0; l[i]=local_given_upload(p,i,0.0); }
        auto loads = server_loads(p,u,a);
        update_edges(p,loads,d);
        const long double initial = objective(p,l,u,d,a);
        long double previous=initial, max_rise=0.0L;
        int sweeps=0, accepted=0, objective_checks=0;
        for (int sweep=0; sweep<max_sweeps; ++sweep) {
            ++sweeps;
            loads=server_loads(p,u,a);
            update_edges(p,loads,d);
            long double current=objective(p,l,u,d,a);
            max_rise=std::max(max_rise,current-previous); previous=current; ++objective_checks;
            int changed=0;
            for (int i=0; i<n; ++i) {
                const int old=a[i];
                if (old>=0) loads[old]=std::max(0.0,loads[old]-u[i]);
                const double old_others = old>=0 ? p.h[old]-d[old]+loads[old] : 0.0;
                const double old_cost = device_cost(p,i,old,l[i],u[i],old_others);
                Candidate best = minimize_server(p,i,-1,0.0);
                for (int j=0; j<m; ++j) {
                    const auto candidate=minimize_server(p,i,j,p.h[j]-d[j]+loads[j]);
                    // Enumeration starts with null; exact ties prefer null,
                    // then the smallest server index. Tiny numerical changes
                    // still require the strict acceptance check below.
                    if (candidate.cost < best.cost) best=candidate;
                }
                if (old_cost-best.cost > tolerance) {
                    a[i] = best.upload>0.0 ? best.server : -1;
                    l[i]=best.local; u[i]=best.upload;
                    ++changed; ++accepted;
                    current=objective(p,l,u,d,a);
                    max_rise=std::max(max_rise,current-previous); previous=current; ++objective_checks;
                }
                if (a[i]>=0) loads[a[i]]+=u[i];
            }
            if (changed==0) break;
        }
        loads=server_loads(p,u,a);
        update_edges(p,loads,d);
        const long double final=objective(p,l,u,d,a);
        max_rise=std::max(max_rise,final-previous); ++objective_checks;
        // A fresh full unilateral scan distinguishes convergence from merely
        // exhausting the sweep budget. Edge blocks are optimal after the last update.
        double best_deviation=0.0;
        for (int i=0; i<n; ++i) {
            const int old=a[i];
            if (old>=0) loads[old]=std::max(0.0,loads[old]-u[i]);
            const double old_others=old>=0 ? p.h[old]-d[old]+loads[old] : 0.0;
            const double old_cost=device_cost(p,i,old,l[i],u[i],old_others);
            double best=minimize_server(p,i,-1,0.0).cost;
            for (int j=0; j<m; ++j)
                best=std::min(best,minimize_server(p,i,j,p.h[j]-d[j]+loads[j]).cost);
            best_deviation=std::max(best_deviation,old_cost-best);
            if (old>=0) loads[old]+=u[i];
        }
        statistics[0]=(double)initial;
        statistics[1]=(double)final;
        statistics[2]=(double)max_rise;
        statistics[3]=sweeps;
        statistics[4]=accepted;
        statistics[5]=best_deviation;
        statistics[6]=best_deviation<=tolerance ? 1.0 : 0.0;
        statistics[7]=objective_checks;
        return 0;
    } catch (...) { return 1; }
}
