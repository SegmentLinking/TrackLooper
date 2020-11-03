from ROOT import *

class Reader(object):
    def __init__(self, fname=None, treename="Events"):
        self.nevents = 0
        self.ievent = 0
        self.alias_to_branchname = { }
        self.alias_to_classname = { }
        self.alias_to_isvector = { }

        self.treename = treename
        self.ch = TChain(self.treename)

        if fname: 
            self.ch.Add(fname)
            self.make_alias_map()

    def get_ievent(self):
        return self.ievent

    def __repr__(self):
        return "<Event %i/%i of tree %s>" % (self.ievent, self.nevents, self.treename)

    def add_file(self, fname):
        self.ch.Add(fname)
        self.make_alias_map()

    def get_list_of_files(self):
        filenames = [f.GetTitle() for f in self.ch.GetListOfFiles()]
        return filenames

    def get_list_of_branches(self):
        return list(self.alias_to_branchname.keys())

    def make_alias_map(self):
        filenames = self.get_list_of_files()
        f1 = TFile(filenames[0])
        tree = f1.Get(self.treename)
        aliases = tree.GetListOfAliases()
        branches = tree.GetListOfBranches()

        if not not aliases:
            for ialias, alias in enumerate(aliases):
                aliasname = alias.GetName()
                branch = tree.GetBranch(self.ch.GetAlias(aliasname))
                branchname = branch.GetName()
                classname = branch.GetClassName()

                self.alias_to_branchname[aliasname] = branchname.replace(".obj","")
                self.alias_to_classname[aliasname] = classname
                self.alias_to_isvector[aliasname] = "vector" in classname

        else:
            for ibranch, branch in enumerate(branches):
                branchname = branch.GetName()
                classname = branch.GetClassName()

                self.alias_to_classname[branchname] = classname
                self.alias_to_branchname[branchname] = branchname.replace(".obj","")
                self.alias_to_isvector[branchname] = "vector<" in classname

    def __getattr__(self, alias):
        # enable branch if this is the first time reading it
        branchname = self.alias_to_branchname[alias]
        classname = self.alias_to_classname[alias]
        val = self.ch.__getattr__(branchname)
        is_wrapper = "edm::Wrapper" in classname

        if is_wrapper: val = val.product()

        if self.alias_to_isvector[alias]:
            return val
        else:
            # wtf? pyroot implicitly stores single variables in vectors, so pick it out
            if is_wrapper:
                return val[0]
            else:
                return val

    def __iter__(self):
        self.ievent = 0
        self.nevents = self.ch.GetEntries()
        return self

    def next(self):
        if self.ievent >= self.nevents:
            raise StopIteration
        else:
            self.ch.GetEntry(self.ievent)
            self.ievent  += 1
            return self

if __name__ == "__main__":
    fname = "ntuple.root"
    reader = Reader(fname, treename="Events")

    for event in reader:
        print event.evt_event, event.evt_run, event.evt_lumiBlock
        print map(lambda x: x.Pt(), event.mus_p4)
        
