#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "icm_gpu.h"
#define N_REPS 5
#define Q_POINTS 256
static void make_stacks(int n, std::vector<double> &S){S.resize(n);srand(123+n);for(int i=0;i<n;++i)S[i]=1.0+99.0*((double)rand()/RAND_MAX);}
static void make_payout(int n,int k,std::vector<double> &p){p.resize(k);for(int m=0;m<k;++m)p[m]=(double)(n-m);}
static double median_of(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
struct Point{int n,k;const char*tag;};
static const Point pts[] = {
  {524288,524288,"anchor B=64"},
  {650000,650000,"gap"},
  {741455,741455,"gap midpoint(524288/1048576)"},
  {850000,850000,"gap"},
  {1000000,1000000,"gap near 1048576"},
  {1048576,1048576,"anchor B=32"},
  {524288,100,"anchor k100 B=64"},
  {650000,100,"gap k100"},
  {741455,100,"gap midpoint k100"},
  {850000,100,"gap k100"},
  {1000000,100,"gap k100"},
  {1048576,100,"anchor k100 B=32"},
};
int main(){
  if(!icm_gpu_init(0)){fprintf(stderr,"init fail\n");return 1;}
  printf("%-10s%-10s%-30s%-10s\n","n","k","tag","median_ms");
  for (auto &pt : pts) {
    int n=pt.n,k=pt.k;
    std::vector<double> S,payout,eq; make_stacks(n,S); make_payout(n,k,payout); eq.assign(n,0.0);
    IcmGpuOptions opts{}; opts.device_id=0; opts.use_cufftdx=1; opts.enable_graphs=0; opts.enable_q_pipeline=1; opts.memory_strategy=0; opts.force_uncached_fused_levels=-1; opts.force_uncached_cufft_levels=-1;
    IcmGpuPlan *plan = icm_gpu_plan_create(n, S.data(), k, &opts);
    if(!plan){printf("%-10d%-10d%-30sPLAN_FAIL\n",n,k,pt.tag);continue;}
    IcmGpuRunStats warm{}; icm_gpu_equity_with_plan(plan,Q_POINTS,payout.data(),eq.data(),&warm);
    std::vector<double> samples;
    for(int r=0;r<N_REPS;++r){IcmGpuRunStats st{}; if(icm_gpu_equity_with_plan(plan,Q_POINTS,payout.data(),eq.data(),&st)!=0)break; samples.push_back(st.total_ns/1e6);}
    icm_gpu_plan_destroy(plan); icm_gpu_release_pooled_memory();
    if(samples.size()<N_REPS){printf("%-10d%-10d%-30sINCOMPLETE\n",n,k,pt.tag);continue;}
    printf("%-10d%-10d%-30s%-10.3f\n",n,k,pt.tag,median_of(samples));
    fflush(stdout);
  }
  icm_gpu_shutdown();
  return 0;
}
