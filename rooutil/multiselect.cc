#include "multiselect.h"
#include <TTreeFormula.h>

Bool_t TSelectorMultiDraw::CompileVariables(const char *varexp/* = ""*/, const char *selection/* = ""*/) {
    Bool_t ret = TSelectorDraw::CompileVariables(varexp, selection);

    // Disable quick load on all formulas
    if (fSelect)
        fSelect->SetQuickLoad(false);

    for (size_t i = 0; i < (size_t) fDimension; i++) {
        fVar[i]->SetQuickLoad(false);
    }

    return ret;
}
