#include "QFramework/TQLibrary.h"

#include "QFramework/TQCutflowPlotter.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQIterator.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQCutflowPlotter:
//
// The TQCutflowPlotter class uses an TQSampleDataReader to obtain
// cutflow numbers to plot a cutflow as a bar-chart.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQCutflowPlotter)

TQCutflowPlotter::TQCutflowPlotter(TQSampleFolder* sf) : TQPresenter(sf) {
  // constructor using a sample folder
}

TQCutflowPlotter::TQCutflowPlotter(TQSampleDataReader* reader) : TQPresenter(reader) {
  // constructor using a sample data reader
}



bool TQCutflowPlotter::writeToFile(const TString& filename, const TString& tags){
  // write the cutflow plot to a file
  // supported formats (can be set with "format=<format>") are:
  // - tikz: TikZ-formatted LaTeX text file for a bar chart
  // - plain: CSV-formatted numbers for processing with some external software
  TQTaggable tmp(tags);
  return this->writeToFile(filename,tmp);
}

bool TQCutflowPlotter::writeToFile(const TString& filename, TQTaggable& tags){
  // write the cutflow plot to a file
  // supported formats (can be set with "format=<format>") are:
  // - tikz: TikZ-formatted LaTeX text file for a bar chart
  // - plain: CSV-formatted numbers for processing with some external software
  tags.importTags(this);
  std::ofstream ofile(filename);
  if(!ofile.is_open()){
    return false;
  }
  TString format = "tikz";
  //@tag: [format] Sets the format in which the cut flow is plotted. Default: "tikz", other values: "plain". Argument tag overrides object tag.
  tags.getTagString("format",format);
  format.ToLower();
  if(format.CompareTo("tikz") == 0){
    this->writeTikZHead(ofile,tags);
    this->writeTikZBody(ofile,tags);
    this->writeTikZFoot(ofile,tags);
  } else if(format.CompareTo("plain") == 0){
    this->writePlain(ofile,tags);
  } else {
    ofile.close();
    return false;
  }
  ofile.close();
  return true;
}


void TQCutflowPlotter::writeTikZHead(std::ostream& out, TQTaggable& tags){
  // write the TikZ-head to a stream
  tags.importTags(this);
  //@tag: [standalone] If this argument tag is set to true, tikz formated output is produced as a standalone version. Argument tag overrides object tag.
  if(tags.getTagBoolDefault("standalone",true)){
    out << "\\documentclass {standalone}" << std::endl;
    out << "\\usepackage{pgfplots, pgfplotstable}" << std::endl;
    out << "\\pgfplotsset{compat=1.8} " << std::endl;
    out << "\\pgfplotsset{single xbar legend/.style={legend image code/.code={\\draw[##1,/tikz/.cd,bar width=6pt,bar shift=0pt,xbar] plot coordinates {(0.8em,0pt)};}}}" << std::endl;
    out << "\\begin{document}" << std::endl;
  }
}

void TQCutflowPlotter::writePlain(std::ostream& out, TQTaggable& tags){
  // write a the "plain" numbers in CSV format to some stream
  tags.importTags(this);  
  //@tag:[style.plain.sep] This tag specifies the seperator used in plain text output. Default: ",". Argument tag overrides object tag.
  TString sep = tags.getTagStringDefault("style.plain.sep",",");
  //@tag:[style.plain.leadCellContent] This tag specifies the content of the top left cell in plain text output. Default: "Cut/Process". Argument tag overrides object tag.
  out << tags.getTagStringDefault("style.plain.leadCellContent","Cut/Process");
  TQTaggableIterator processes(this->fProcesses);
  while(processes.hasNext()){
    TQTaggable* process = processes.readNext();
    if(process){
      TString processPath;
      //@tag: [.path] This process tag specifies the location of the process in the TQFolder structure.
      if(process->getTagString(".path",processPath)){
				processPath = tags.replaceInText(processPath);
				if(TQFolder::isValidPath(processPath,true,true,true)){
					out << sep << processPath;
				}
      }
    }
  }
  out << std::endl;
  TQTaggableIterator cuts(this->fCuts);
  while(cuts.hasNext()){
    TQTaggable* cut = cuts.readNext();
    if(!cut) continue;
    TString cutTitle, cutName;
    //@tag: [.name] This cut/process tag specifies the name of the respective cut/process
    if(cut->getTagString(".name",cutName)){
      if(cutName.BeginsWith("|")) continue;
      out << cutName;
      std::vector<double> numbers;
      processes.reset(); 
      while(processes.hasNext()){
        TQTaggable* process = processes.readNext();
        if(process){
          TString processPath;
          if(process->getTagString(".path",processPath)){
						processPath = tags.replaceInText(processPath);
						if(TQFolder::isValidPath(processPath,true,true,true)){
							TQCounter * counter = fReader->getCounter(processPath, cutName,&tags);
							if (counter) {
								double number = counter->getCounter();
								if(TQUtils::isNum(number)) numbers.push_back(number);
								else numbers.push_back(0.);
								delete counter;
							} else {
								numbers.push_back(0.);
							}
						}
					}
        }
      }
      //@tag: [style.normalize] If this tag is set to true, the cutflow is normalized such that the sum of all processes at one cut stage is one. Default: false. Argument tag overrides object tag.
      if(tags.getTagBoolDefault("style.normalize",false)){
        double normalization = 0.;
        for(size_t i=0; i<numbers.size(); i++){
          normalization += numbers[i];
        }
        if(normalization > 0){
          for(size_t i=0; i<numbers.size(); i++){
            out << sep;
            out << numbers[i]/normalization;
          }
        } else {
          for(size_t i=0; i<numbers.size(); i++){
            out << sep;
            out << 0;
          }
        }
      } else {
        for(size_t i=0; i<numbers.size(); i++){
          out << sep;
          out << numbers[i];
        }
      }
      out << std::endl;
    }
  }
}

void TQCutflowPlotter::writeTikZBody(std::ostream& out, TQTaggable& tags){
  // write the TikZ-body to a stream

  tags.importTags(this);
  std::vector<TString> styles;
  std::vector<TString> processnames;
  std::vector<TString> processtitles;
  TQTaggableIterator processes(this->fProcesses);
	if(!processes.hasNext()){
		WARNclass("no processes booked, writing empty cutflow plot!");
	}
  while(processes.hasNext()){
    TQTaggable* process = processes.readNext();
    if(process){
      TString processPath, processName, processStyle, processTitle;
			if(process->getTagString(".path",processPath)){
				processPath = tags.replaceInText(processPath);
				if(TQFolder::isValidPath(processPath,true,true,true)){
					if(!process->getTagString(".name",processName)){
						processName = TQFolder::makeValidIdentifier(processPath);
					}
					//@tag: [.style] This process tag specifies the style used in tikz output for the respective process. If it is not set on the process, it is retrieved from the common base folder of the folders belonging to this process.
					TList* folders = fReader->getListOfSampleFolders(processPath);
					TQFolder* f = fReader->getSampleFolder()->findCommonBaseFolder(folders);
					delete folders;
					processStyle = "";
					int color;
					//@tag: [~style.default.histFillColor] This (sample-)folder tag specifies the fill color used for processes for which folder is the common base folder.
					if(process->getTagInteger("histFillColor",color)||process->getTagInteger("color",color)||f->getTagInteger("~style.default.histFillColor",color)){
						processStyle.Append("fill=");
						TString colorName = processName;
						colorName.Prepend("fill");
						TString colordef =  TQStringUtils::getColorDefStringLaTeX(colorName,color);
						out << colordef << std::endl;
						processStyle.Append(colorName);
					}
					//@tag: [~style.default.histLineColor] This (sample-)folder tag specifies the line color used for processes for which folder is the common base folder.
					if(process->getTagInteger("histLineColor",color)||process->getTagInteger("color",color)||f->getTagInteger("~style.default.histLineColor",color)){
						if(color != kBlack){
							if(!processStyle.IsNull()) processStyle.Append(",");
							processStyle.Append("draw=");
							TString colorName = processName;
							colorName.Prepend("draw");
							TString colordef =  TQStringUtils::getColorDefStringLaTeX(colorName,color);
							out << colordef << std::endl;
							processStyle.Append(colorName);
						}
					}
					process->getTagString(".style",processStyle);
					if(f){
						f->getTagString("~style.default.title",processTitle);
					}
					process->getTagString(".title",processTitle);
					styles.push_back(processStyle);
					processnames.push_back(processName);
					processtitles.push_back(TQStringUtils::convertROOTTeX2LaTeX(processTitle));
				}
			}
    }
  }
 
  out << "\\begin{tikzpicture}" << std::endl;
  out << "\\pgfplotstableread[col sep=comma]{" << std::endl;
 
  out << "label";
  for(size_t i=0; i<processnames.size(); i++){
    out << "," << "{" << processnames[i] << "}";
  }
  out << std::endl;
 
  int nCuts = 0;
  TQTaggableIterator cuts(this->fCuts);
  while(cuts.hasNext()){
    TQTaggable* cut = cuts.readNext();
    if(!cut) continue;
    TString cutTitle, cutName;
    if(!cut->getTagString(".name",cutName)) continue;
    if(!cut->getTagString(".title",cutTitle)) cutTitle=cutName;
    if(cutName.BeginsWith("|")) continue;
		cutTitle.ReplaceAll("_","");
    out << "{" << cutTitle << "}"; 
    nCuts++;
    std::vector<double> numbers;
    processes.reset(); 
    while(processes.hasNext()){
      TQTaggable* process = processes.readNext();
      if(process){
        TString processPath;
				if(process->getTagString(".path",processPath)){
					processPath = tags.replaceInText(processPath);
					if(TQFolder::isValidPath(processPath,true,true,true)){
						TQCounter * counter = fReader->getCounter(processPath, cutName,&tags);
						if (counter) {
							double number = counter->getCounter();
							if(TQUtils::isNum(number)) numbers.push_back(number);
							else numbers.push_back(0.);
							delete counter;
						} else {
							numbers.push_back(0.);
						}
					}
        }
      }
    }
    if(tags.getTagBoolDefault("style.normalize",true)){
      double normalization = 0.;
      for(size_t i=0; i<numbers.size(); i++){
        normalization += numbers[i];
      }
      if(normalization > 0){
        for(size_t i=0; i<numbers.size(); i++){
          out << ",";
          out << "{" << numbers[i]/normalization << "}";
        }
      } else {
        for(size_t i=0; i<numbers.size(); i++){
          out << ",";
          out << "{" << 0 << "}";
        }
      }
    } else {
      for(size_t i=0; i<numbers.size(); i++){
        out << ",";
        out << "{" << numbers[i] << "}";
      }
    }
    out << std::endl;
  }
 
  out << "}\\datatable" << std::endl;
  out << "\\begin{axis}[" << std::endl;
  //@tag: [style.tikz.lineHeight] This tag determines the line height in tikz formated output. Default: "1.5em". Argument tag overrides object tag. 
  out << " height=" << nCuts << "*" << tags.getTagStringDefault("style.tikz.lineHeight","1.5em") << "," << std::endl;
  //@tag: [style.tikz.width] This tag determines the width in tikz formated output. Default: "3cm". Argument tag overrides object tag. 
  out << " width=" << tags.getTagStringDefault("style.tikz.width","3cm") << "," << std::endl;
  out << " scale only axis," << std::endl;
  //@tag: [style.tikz.enlargeLimits] This tag determines value of the "enlarge y limits" option in tikz formated output. Default: "5pt". Argument tag overrides object tag. 
  out << " enlarge y limits={abs=" << tags.getTagStringDefault("style.tikz.enlargeLimits","5pt") << "}," << std::endl;
  out << " xmin=0," << std::endl;
  out << " tick align=outside," << std::endl;
  out << " axis x line*=bottom," << std::endl;
  out << " axis y line*=left," << std::endl;
	out << " legend cell align=left,legend style={draw=none,at={(1.10,0.5)},anchor=west}," << std::endl;
  out << " every y tick/.style={draw=none}," << std::endl;
  out << " xbar stacked, " << std::endl;
  //@tag: [style.tikz.reverseYordering] If this tag is set to true, the option "y dir=reverse" is added for tikz formated output. Default: true. Argument tag overrides object tag.
  if(tags.getTagBoolDefault("style.tikz.reverseYordering",true)) out << " y dir=reverse," << std::endl; 
  //@tag: [style.tikz.normalize] If this tag is set to true, the following options are added for tikz formated output: "scaled x ticks=false", "x tick label style={/pgf/number format/fixed}", "enlarge x limits=false". If set to false, "enlarge x limits=upper" is used. Default: true. Argument tag overrides object tag.
  if(tags.getTagBoolDefault("style.tikz.normalize",true)){
    out << " scaled x ticks=false," << std::endl; 
    out << " x tick label style={/pgf/number format/fixed}," << std::endl;
    out << " enlarge x limits=false," << std::endl;
  } else {
    out << " enlarge x limits=upper," << std::endl;
  }
  //@tag: [style.tikz.barWidth] This tag specifies the bar width for tikz formated output. Default: "10pt". Argument tag overrides object tag.
  out << " bar width=" << tags.getTagStringDefault("style.tikz.barWidth","10pt") << "," << std::endl;
  out << " ytick=data," << std::endl;
  out << " yticklabels from table={\\datatable}{label}," << std::endl;
  out << "]" << std::endl;
 
  processes.reset();
  size_t idx = 0;
  while(processes.hasNext()){
    TQTaggable* process = processes.readNext();
    if(process){
      TString processStyle, processName, processPath, processTitle;
      if(process->getTagString(".path",processPath)){
				processPath = tags.replaceInText(processPath);
				if(TQFolder::isValidPath(processPath,true,true,true)){
					if(!process->getTagString(".name",processName)){
						processName = TQFolder::makeValidIdentifier(processPath);
					}
					out << "\\addplot[";
					out << styles[idx];
					out << "] table [x=" << processName << ", y expr=\\coordindex] {\\datatable};" << std::endl;
					idx++;
				}
      }
    }
  }
  //@tag: [style.showLegend] If this tag is set to true, a legend is added to tikz formated output. Default: false. Argument tag overrides object tag.
  if(tags.getTagBoolDefault("style.showLegend",false)){
		bool mathtitles = tags.getTagBoolDefault("style.legendMathTitles",false);
    out << "\\legend{";
    for(size_t i=0; i<processtitles.size(); i++){
      if(i!=0) out << ",";
			if(mathtitles || ( (processtitles[i].Contains("\\") || processtitles[i].Contains("_") || processtitles[i].Contains("^")) && !processtitles[i].Contains("$")) ){
				out << "\\ensuremath{" << processtitles[i] << "}";
			} else {
				out << processtitles[i];
			}
    }
    out << "}" << std::endl;
  }
  out << "\\end{axis}" << std::endl;
  out << "\\end{tikzpicture}%" << std::endl;
 
}

void TQCutflowPlotter::writeTikZFoot(std::ostream& out, TQTaggable& tags){
  // write the TikZ-foot to a stream
  if(tags.getTagBoolDefault("standalone",true)){
    out << "\\end{document}" << std::endl;
  }
}

void TQCutflowPlotter::setup(){
  // set some default tags
}

