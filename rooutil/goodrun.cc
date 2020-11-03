// $Id: goodrun.cc,v 1.12 2012/11/12 18:58:08 kelley Exp $

// CINT is allowed to see this, but nothing else:
#include "goodrun.h"

#ifndef __CINT__

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <set>
#include <string>

enum file_type { TEXT, JSON };

struct run_and_lumi {
     unsigned int run;
     long long int lumi_min;
     long long int lumi_max;
};

bool operator < (const struct run_and_lumi &r1, const struct run_and_lumi &r2)
{
     return r1.run < r2.run;
}

typedef std::multiset<struct run_and_lumi> set_t; 
static set_t good_runs_;
static bool good_runs_loaded_ = false;

static const char json_py[] =
"#! /usr/bin/env python                                                                                       \n"
"													      \n"
"# usage to print out good run selection for goodrun.cc:						      \n"
"# convertGoodRunsList_JSON.py <json file>								      \n"
"													      \n"
"import sys,json											      \n"
"													      \n"
"runs = json.load(open(sys.argv[1],\"r\"))								      \n"
"													      \n"
"for run in runs.keys():										      \n"
"    for lumiBlock in runs[run] :									      \n"
"        if len(lumiBlock) != 2 :									      \n"
"            print 'ERROR reading lumi block: run:',run,'lumiBlock:',lumiBlock				      \n"
"        else:												      \n"
"            # print 'run ' + str(run) + ', min lumi ' + str(lumiBlock[0]) + ', max lumi ' + str(lumiBlock[1])\n"
"            print run,lumiBlock[0],lumiBlock[1]							      \n"
     "													      \n";
//"print ''                                                                                                     \n";

static int load_runs (const char *fname, enum file_type type)
{
     good_runs_.clear();
     FILE *file = 0;
     switch (type) { 
     case TEXT:
	  file = fopen(fname, "r");
	  if (file == 0) {
	       perror("opening good run list");
	       return 0;
	  }
	  break;
     case JSON:
     {
	  // first make a temp file with the python parsing code
	  FILE *pyfile = fopen("tmp.py", "w");
	  if (pyfile == 0) {
	       perror("opening tmp file");
	       return 0;
	  }
	  fprintf(pyfile, "%s", json_py);
	  fclose(pyfile);
	  chmod("tmp.py", S_IRUSR | S_IWUSR | S_IXUSR);
	  // now execute a command to convert the JSON file to text
	  pid_t child = fork();
	  if (child == -1) {
	       perror("forking process to convert JSON good run list");
	       return 0;
	  }
	  if (child == 0) {
	       FILE *f = freopen((std::string(fname) + ".tmp").c_str(), "w", stdout);
	       if (f == 0) {
		    perror("opening good run list");
		    return 127;
	       }
	       execlp("./tmp.py", "", fname, (char *)0);
	       perror("executing JSON conversion script");
	       exit(127);
	  }
	  if (child != 0) {
	       int status;
	       waitpid(child, &status, 0);
	       if (status != 0) {
		    printf("conversion exited abnormally, consult previous errors\n");
		    return 0;
	       }
	  }
	  printf("note: converted JSON file is in %s.tmp, "
		 "please consult in case of parsing errors\n", fname);
	  file = fopen((std::string(fname) + ".tmp").c_str(), "r");
	  if (file == 0) {
	       perror("opening good run list");
	       return 0;
	  }
	  unlink("tmp.py");
	  break;
     }
     default:
	  break;
     }
     int s;
     int line = 0;
     do {
	  int n;
	  char buf[1024] = "";
	  // read a line from the file, not including the newline (if
	  // there is a newline)
	  s = fscanf(file, "%1024[^\n]%n", buf, &n);
	  assert(n < 1023);
	  if (s != 1) {
	       if (s != EOF) {
		    perror("reading good run list");
		    return 0;
	       } else {
		    if (ferror(file)) {
			 perror("reading good run list");
			 return 0;
		    }
	       }
	  } else if (strlen(buf) != 0 && buf[0] == '#') {
	       line++;
	       // printf("Read a comment line (line %d) from the good run list: %s\n", line, buf);
	  } else {
	       line++;
	       // printf("Read a line from the good run list: %s\n", buf);
	       unsigned int run;
	       char *pbuf = buf;
		   s = sscanf(pbuf, " %u%n", &run, &n);
	       if (s != 1) {
		    fprintf(stderr, "goodrun: Expected a run number (unsigned int)"
			    " in the first position of line %d: %s\n", line, buf);
		    assert(s == 1);
	       }
	       pbuf += n;
	       long long int lumi_min = -1;
	       long long int lumi_max = -1;
	       s = sscanf(pbuf, " %lld%n", &lumi_min, &n);
	       // if there is no lumi_min specified, that means the
	       // entire run is good
	       if (s == 1) {
		    pbuf += n;
		    s = sscanf(pbuf, " %lld%n", &lumi_max, &n);
		    if (s != 1) {
			 fprintf(stderr, "goodrun: Expected a max lumi section in a lumi section range"
				 " (int) in the third position of line %d: %s\n", line, buf);
			 assert(s == 1);
		    }
		    pbuf += n;
	       }
	       char trail[1024] = "";
	       s = sscanf(pbuf, " %s", trail);
	       if (strlen(trail) != 0) {
		    fprintf(stderr, "goodrun: Unexpected trailing junk (%s) on line %d: %s\n", trail, line, buf);
		    assert(s == 0);
	       }
	       // printf("Read line: run %u, min lumi %lld, max lumi %lld\n", run, lumi_min, lumi_max);
	       struct run_and_lumi new_entry = { run, lumi_min, lumi_max };
	       good_runs_.insert(new_entry);
	  }
	  // advance past the newline 
	  char newlines[1024] = "";
	  s = fscanf(file, "%[ \f\n\r\t\v]", newlines); 
	  if (s != -1 && strlen(newlines) != 1) {
		fprintf(stderr, "goodrun: Warning: unexpected white space following line %d\n", line);
		// but that's just a warning
	  } 
     } while (s == 1);
     fclose(file);
     if (type == JSON)
	  unlink((std::string(fname) + ".tmp").c_str());
     return line;
}

bool goodrun (unsigned int run, unsigned int lumi_block)
{
     if (not good_runs_loaded_) {
	  int ret = load_runs("goodruns.txt", TEXT);
	  assert(ret != 0);
	  good_runs_loaded_ = true;
     }
     // we assume that an empty list means accept anything
     if (good_runs_.size() == 0)
	  return true;
     // find all blocks with this run number
     struct run_and_lumi r_a_l = { run, 0, 0 };
     std::pair<set_t::const_iterator, set_t::const_iterator> good_blocks = 
	  good_runs_.equal_range(r_a_l);
     for (set_t::const_iterator i = good_blocks.first; 
	  i != good_blocks.second; ++i) {
// 	  printf("considering run %u, min %lld, max %lld\n", 
// 		 i->run, i->lumi_min, i->lumi_max);
	  if (i->lumi_min <= lumi_block) {
	       if (i->lumi_max == -1 || i->lumi_max >= lumi_block)
		    return true;
	  }
     }
     return false;
}

int min_run ()
{
     if (not good_runs_loaded_)
	  return -1;
     set_t::const_iterator first = good_runs_.begin();
     if (first != good_runs_.end())
	  return first->run;
     return -1;
}

int max_run ()
{
     if (not good_runs_loaded_)
	  return -1;
     set_t::const_iterator last = good_runs_.end();
     if (last != good_runs_.begin()) {
	  last--;
	  return last->run;
     }
     return -1;
}

bool goodrun_json (unsigned int run, unsigned int lumi_block)
{
     if (not good_runs_loaded_) {
	  int ret = load_runs("goodruns.json", JSON);
	  assert(ret != 0);
	  good_runs_loaded_ = true;
     }
     // once the JSON good-run list is loaded, there's no difference
     // between TEXT and JSON
     return goodrun(run, lumi_block);
}

void set_goodrun_file (const char* filename)
{
  int ret = load_runs(filename, TEXT);
  assert(ret != 0);
  good_runs_loaded_ = true;
}

void set_goodrun_file_json (const char* filename)
{
     int ret = load_runs(filename, JSON);
     assert(ret != 0);
     good_runs_loaded_ = true;
}

int min_run_min_lumi ()
{
     if (not good_runs_loaded_)
      return -1;
     set_t::const_iterator first = good_runs_.begin();
     if (first != good_runs_.end())
      return first->lumi_min;
     return -1;
}

int max_run_max_lumi ()
{
     if (not good_runs_loaded_)
      return -1;
     set_t::const_iterator last = good_runs_.end();
     if (last != good_runs_.begin()) {
      last--;
      return last->lumi_max;
     }
     return -1;
}

#endif // __CINT__

