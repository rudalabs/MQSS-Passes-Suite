include "mlir/Pass/PassBase.td"

def ReverseCx : Pass<"ReverseCx"> {
  let summary = "Decomposition pass that reverses the control and targets of "
                "each two-qubits Cx gate in a circuit";
  let constructor = "mqss::opt::createReverseCxPass()";
}

def CxToLowerHCzH : Pass<"CxToLowerHCzH"> {
  let summary = "Decomposition pass of Cx by H, Cz, and H";
  let constructor = "mqss::opt::createCxToLowerHCzHPass()";
}

def CxToUpperHCzH : Pass<"CxToUpperHCzH"> {
  let summary = "Decomposition pass of Cx by H, Cz, and H";
  let constructor = "mqss::opt::createCxToUpperHCzHPass()";
}

def CzToLowerHCxH : Pass<"CzToLowerHCxH"> {
  let summary = "Decomposition pass of Cz by H, Cx, and H";
  let constructor = "mqss::opt::createCzToLowerHCxHPass()";
}

def CzToUpperHCxH : Pass<"CzToUpperHCxH"> {
  let summary = "Decomposition pass of Cz by H, Cx, and H";
  let constructor = "mqss::opt::createCzToUpperHCxHPass()";
}

def RxToHRzH : Pass<"RxToHRzH"> {
  let summary = "Decomposition pass of Rx by H, Rz and H";
  let constructor = "mqss::opt::createRxToHRzHPass()";
}

def RzToHRxH : Pass<"RzToHRxH"> {
  let summary = "Decomposition pass of Rz by H, Rx, and H";
  let constructor = "mqss::opt::createRzToHRxHPass()";
}


def CrxToHCrzH : Pass<"CrxToHCrzH"> {
  let summary = "Decomposition pass of Crx by H, Crz and H";
  let constructor = "mqss::opt::createCrxToHCrzHPass()";
}

def CrzToHCrxH : Pass<"CrzToHCrxH"> {
  let summary = "Decomposition pass of Crz by H, Crx, and H";
  let constructor = "mqss::opt::createCrzToHCrxHPass()";
}
def SdgToSSS : Pass<"SdgToSSS"> {
  let summary = "Decomposition pass that replaces a Sdg gate by S S S gates";
  let constructor = "mqss::opt::createSdgToSSSPass()";
}

def SToSdgSdgSdg : Pass<"SToSdgSdgSdg"> {
  let summary = "Decomposition pass that replaces an S gate by Sdg Sdg Sdg";
  let constructor = "mqss::opt::createSToSdgSdgSdgPass()";
}

def SToTT : Pass<"SToTT"> {
  let summary = "Decomposition pass that replaces S by T T gates";
  let constructor = "mqss::opt::createSToTTPass()";
}

def XToHZH : Pass<"XToHZH"> {
  let summary = "Decomposition pass of X by H, Z and H";
  let constructor = "mqss::opt::createXToHZHPass()";
}

def ZToHXH : Pass<"ZToHXH"> {
  let summary = "Decomposition pass of Z by H, X and H";
  let constructor = "mqss::opt::createZToHXHPass()";
}

def SwapToLowerCxCxCx : Pass<"SwapToLowerCxCxCx"> {
  let summary = "Decomposition pass of Swap by Cx Cx Cx gates";
  let constructor = "mqss::opt::createSwapToLowerCxCxCxPass()";
}

def SwapToUpperCxCxCx : Pass<"SwapToUpperCxCxCx"> {
  let summary = "Decomposition pass of Swap by Cx Cx Cx gates";
  let constructor = "mqss::opt::createSwapToUpperCxCxCxPass()";
}

