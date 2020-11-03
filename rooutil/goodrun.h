
#ifndef GOODRUN_H
#define GOODRUN_H

bool goodrun (unsigned int run, unsigned int lumi_block);
bool goodrun_json (unsigned int run, unsigned int lumi_block);
void set_goodrun_file (const char* filename);
void set_goodrun_file_json (const char* filename);
int  min_run ();
int  max_run ();
int  min_run_min_lumi ();
int  max_run_max_lumi ();

#endif

