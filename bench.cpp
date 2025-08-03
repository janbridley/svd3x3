// NOTE: requires ulimit -s 65500 on macos, otherwise we stack overflow
// I could use a standard vector, but I want the most accurate benchmarks so the
// additional cast/heap access are undesirable
#include <iostream>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>

#define NS_TO_US 0.001
#define N_SAMPLES 500000

// #ifdef BENCH_ORIGINAL
// #include "extern/svd_prev/svd3.h"
// #else
#include "svd3.h"

#include <random>

void setup(double (&samples)[N_SAMPLES][9]) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-100.0, 100.0);

  for (int x = 0; x < N_SAMPLES; ++x)
    for (int i = 0; i < 9; ++i)
      samples[x][i] = dist(gen);
}

void bench(double a11, double a12, double a13, double a21, double a22,
           double a23, double a31, double a32, double a33) {
  double m[3][3] = {a11, a12, a13, a21, a22, a23, a31, a32, a33};
  double u[3][3], s[3][3], v[3][3];
  svd(m, u, s, v);
}

// #endif

double compute_mean(const std::vector<double> &data) {
  double sum = 0;
  for (auto d : data)
    sum += d;
  return sum / data.size();
}

double compute_stdev(const std::vector<double> &data, double mean) {
  double sq_sum = 0;
  for (auto d : data)
    sq_sum += (d - mean) * (d - mean);
  return std::sqrt(sq_sum / data.size());
}

int main() {
  // Attempt to lock down to a single thread
  thread_port_t thread = pthread_mach_thread_np(pthread_self());
  thread_affinity_policy_data_t policy = {0}; // Bind to core 0
  thread_policy_set(thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
                    1);

  double samples[N_SAMPLES][9];
  setup(samples);

  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  std::vector<double> timings;

  for (int iter = 0; iter < 20; ++iter) {
    uint64_t start = mach_absolute_time();
    for (int i = 0; i < N_SAMPLES; i++) {
      bench(samples[i][0], samples[i][1], samples[i][2], samples[i][3],
            samples[i][4], samples[i][5], samples[i][6], samples[i][7],
            samples[i][8]);
    }
    uint64_t end = mach_absolute_time();

    uint64_t elapsed = end - start;
    uint64_t elapsed_ns = elapsed * info.numer / info.denom;

    double time_per_svd_us = elapsed_ns * NS_TO_US / N_SAMPLES;
    timings.push_back(time_per_svd_us);
  }
  double mean = compute_mean(timings);
  double stdev = compute_stdev(timings, mean);

  std::cout << "Mean time per SVD: " << mean << " μs\n";
  std::cout << "Stddev time per SVD: " << stdev << " μs\n";

  return 0;
}
