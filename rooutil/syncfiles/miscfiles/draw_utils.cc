#include <algorithm>
#include <set>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <numeric>


struct RLE {
    unsigned int run;
    unsigned long long event;
};
std::set<RLE> already_seen;
bool operator ==(const RLE& lhs, const RLE& rhs) {
    if (lhs.run != rhs.run) return false;
    if (lhs.event != rhs.event) return false;
    return true;
}
bool operator <(const RLE& lhs, const RLE& rhs) {
    if (lhs.run != rhs.run) return lhs.run < rhs.run;
    if (lhs.event != rhs.event) return lhs.event < rhs.event;
    return false;
}
bool is_duplicate(unsigned int run, unsigned long long event) {
    return !already_seen.insert({run,event}).second;
}
void clear_list() {
    already_seen.clear();
}

std::map <unsigned int, std::set<unsigned long long> > event_map;
bool is_duplicate_event(unsigned int run, unsigned long long event) {
    if(event_map.count(run) > 0) {
        if(event_map[run].count(event) > 0) {
            return true;
        }
    }
    event_map[run].insert(event);
    return false;
}

std::chrono::time_point<std::chrono::system_clock> t_first = std::chrono::system_clock::now();
std::chrono::time_point<std::chrono::system_clock> t_old = std::chrono::system_clock::now();
std::vector<double> deq;
int smoothing = 40;
int progress(unsigned int ientry, long int nentries) {
    int period = nentries / 3000;

    if(ientry%period == 0) {
        auto now = std::chrono::system_clock::now();
        double dt = ((std::chrono::duration<double>)(now - t_old)).count();
        t_old = now;
        if (deq.size() >= smoothing) deq.erase(deq.begin());
        deq.push_back(dt);
        double avgdt = std::accumulate(deq.begin(),deq.end(),0.)/deq.size();
        float prate = (float)period/avgdt;
        float peta = (nentries-ientry)/prate;
        if (isatty(1)) {
            float pct = (float)ientry/(nentries*0.01);
            if( ( nentries - ientry ) <= period ) {
                pct = 100.0;
                double dt_total = ((std::chrono::duration<double>)(now - t_first)).count();
                printf("\015\033[32m ---> \033[1m\033[31m%4.1f%% \033[34m [Avg rate: %.2f kHz, Time elapsed: %.0f s] \033[0m\033[32m  <---\033[0m\015 ", pct, nentries/(1000.*dt_total), dt_total);
            } else {
                printf("\015\033[32m ---> \033[1m\033[31m%4.1f%% \033[34m [%.2f kHz, ETA: %.0f s] \033[0m\033[32m  <---\033[0m\015 ", pct, prate/1000.0, peta);
            }
            if( ( nentries - ientry ) > period ) fflush(stdout);
            else std::cout << std::endl;

        }
    }
    return 1;
}

